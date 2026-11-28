// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use datafusion_common::{Result, plan_datafusion_err, plan_err};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use super::lance::validate_receipt_file;
use super::plan::{BackfillPlan, BackfillReceipt, BackfillWorkUnit, dataset_url};

const PLAN_MAGIC: &str = "df.lake.backfill.plan";
const WORK_UNIT_MAGIC: &str = "df.lake.backfill.work_unit";
const RECEIPT_MAGIC: &str = "df.lake.backfill.receipt";
const WIRE_VERSION: u8 = 1;
const MAX_WIRE_BYTES: usize = 16 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope<'a, T> {
    magic: &'a str,
    version: u8,
    payload: T,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundWorkUnit<F> {
    plan_fingerprint: F,
    work_unit: BackfillWorkUnit,
}

impl BackfillPlan {
    /// Encode a plan as a versioned JSON message of at most 16 MiB.
    ///
    /// # Errors
    /// Returns an error if the plan is invalid or exceeds the message limit.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        self.validate()?;
        encode(PLAN_MAGIC, self)
    }

    /// Decode and verify a versioned plan. Plan payloads predating the pinned
    /// fragment layouts (version 1) must be prepared again.
    ///
    /// # Errors
    /// Returns an error for invalid JSON, versions, fingerprints or fragment layouts.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self> {
        let plan: Self = decode(PLAN_MAGIC, bytes)?;
        plan.validate()?;
        Ok(plan)
    }
}

impl BackfillWorkUnit {
    /// Encode a work unit bound to a specific backfill plan.
    ///
    /// # Errors
    /// Returns an error if the unit does not belong to the plan or the message
    /// exceeds 16 MiB.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn to_json_bytes(&self, plan: &BackfillPlan) -> Result<Vec<u8>> {
        plan.validate()?;
        plan.validate_unit(self)?;
        encode(
            WORK_UNIT_MAGIC,
            &BoundWorkUnit {
                plan_fingerprint: plan.plan_fingerprint.as_str(),
                work_unit: *self,
            },
        )
    }

    /// Decode a work unit and validate it against the supplied plan.
    ///
    /// # Errors
    /// Returns an error for an invalid message or a plan or fragment mismatch.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn from_json_bytes(bytes: &[u8], plan: &BackfillPlan) -> Result<Self> {
        let message: BoundWorkUnit<String> = decode(WORK_UNIT_MAGIC, bytes)?;
        plan.validate()?;
        if message.plan_fingerprint != plan.plan_fingerprint {
            return plan_err!("Backfill work unit fingerprint does not match the plan");
        }
        plan.validate_unit(&message.work_unit)?;
        Ok(message.work_unit)
    }
}

impl BackfillReceipt {
    /// Encode a receipt for the supplied backfill plan.
    ///
    /// # Errors
    /// Returns an error if the receipt does not match the plan or the message
    /// exceeds 16 MiB.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn to_json_bytes(&self, plan: &BackfillPlan) -> Result<Vec<u8>> {
        self.validate_wire(plan)?;
        encode(RECEIPT_MAGIC, self)
    }

    /// Decode a receipt and validate its fragment, attempt and artifact metadata.
    /// Artifact contents are checked by [`super::finalize_backfill`].
    ///
    /// # Errors
    /// Returns an error for an invalid message or a receipt that does not match
    /// the supplied plan.
    ///
    /// # Public Only for Internal Use:
    /// This is not a public API and is for internal use only; see [API policy] for details.
    /// [API policy]: https://datafusion.apache.org/contributor-guide/api-health.html#internal-public-apis
    #[doc(hidden)]
    pub fn from_json_bytes(bytes: &[u8], plan: &BackfillPlan) -> Result<Self> {
        let receipt: Self = decode(RECEIPT_MAGIC, bytes)?;
        receipt.validate_wire(plan)?;
        Ok(receipt)
    }

    fn validate_wire(&self, plan: &BackfillPlan) -> Result<()> {
        plan.validate()?;
        self.validate_identity(plan)?;
        plan.validate_unit(&self.work_unit)?;
        validate_receipt_file(plan, self, &dataset_url(plan.dataset_uri())?)
    }
}

fn encode<T: Serialize>(magic: &str, payload: &T) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec(&Envelope {
        magic,
        version: WIRE_VERSION,
        payload,
    })
    .map_err(|error| plan_datafusion_err!("Cannot encode backfill message: {error}"))?;
    check_size(&bytes)?;
    Ok(bytes)
}

fn decode<T: DeserializeOwned>(magic: &str, bytes: &[u8]) -> Result<T> {
    check_size(bytes)?;
    let envelope: Envelope<'_, T> = serde_json::from_slice(bytes)
        .map_err(|error| plan_datafusion_err!("Invalid backfill message: {error}"))?;
    if envelope.magic != magic || envelope.version != WIRE_VERSION {
        return plan_err!("Invalid backfill message magic or envelope version");
    }
    Ok(envelope.payload)
}

fn check_size(bytes: &[u8]) -> Result<()> {
    if bytes.len() > MAX_WIRE_BYTES {
        return plan_err!("Backfill message exceeds the {MAX_WIRE_BYTES}-byte limit");
    }
    Ok(())
}

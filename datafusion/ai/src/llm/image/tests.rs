// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use bytes::Bytes;
use datafusion_common::DataFusionError;

use super::*;

const LIMIT: u64 = 32 << 20;
const PNG: &[u8] = include_bytes!("../../../tests/fixtures/vision.png");
const JPEG: &[u8] = include_bytes!("../../../tests/fixtures/vision.jpeg");
const GIF: &[u8] = include_bytes!("../../../tests/fixtures/vision.gif");
const WEBP: &[u8] = include_bytes!("../../../tests/fixtures/vision.webp");

fn media(bytes: &[u8]) -> ResolvedMedia {
    ResolvedMedia {
        bytes: Bytes::copy_from_slice(bytes),
        mime: None,
        etag: None,
    }
}

fn zlib(bytes: &[u8]) -> Vec<u8> {
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
    encoder.write_all(bytes).unwrap();
    encoder.finish().unwrap()
}

fn png_with_chunks(
    (width, height): (u32, u32),
    color: PngColor,
    depth: u8,
    interlace: u8,
    chunks: &[(&[u8; 4], &[u8])],
) -> Vec<u8> {
    let mut png = PNG_SIGNATURE.to_vec();
    let mut ihdr = [0; 13];
    ihdr[..4].copy_from_slice(&width.to_be_bytes());
    ihdr[4..8].copy_from_slice(&height.to_be_bytes());
    ihdr[8..].copy_from_slice(&[depth, color as u8, 0, 0, interlace]);
    append_png_chunk(&mut png, b"IHDR", &ihdr);
    for (kind, data) in chunks {
        append_png_chunk(&mut png, kind, data);
    }
    append_png_chunk(&mut png, b"IEND", &[]);
    png
}

fn png_with_rows(width: u32, height: u32, color: PngColor, rows: &[u8]) -> Vec<u8> {
    png_with_chunks((width, height), color, 8, 0, &[(b"IDAT", &zlib(rows))])
}

fn assert_exec_error(bytes: &[u8], limit: u64, expected: &str) {
    let error = prepare(
        &mut media(bytes),
        16,
        limit,
        ImageResizeFallback::Passthrough,
    )
    .unwrap_err();
    assert!(
        matches!(error, DataFusionError::Execution(ref detail) if detail.contains(expected)),
        "{error}"
    );
}

type DecodedPng = ((usize, usize), PngColor, Vec<Vec<u8>>);

fn decode(bytes: &[u8]) -> Result<DecodedPng> {
    let ImageDecision::Resize(png) = inspect(bytes, LIMIT)? else {
        panic!("expected a supported PNG");
    };
    let mut rows = png.rows();
    let pixels = (0..png.height)
        .map(|_| rows.read(true).map(<[u8]>::to_vec))
        .collect::<Result<_>>()?;
    rows.finish()?;
    Ok(((png.width, png.height), png.color, pixels))
}

#[test]
fn recognizes_image_formats_and_rejects_invalid_headers() -> Result<()> {
    let mut png = media(PNG);
    let original = png.bytes.as_ptr();
    prepare(&mut png, 33, LIMIT, ImageResizeFallback::Error)?;
    assert_eq!(png.mime.as_deref(), Some(PNG_MIME));
    assert_eq!(png.bytes.as_ref(), PNG);
    assert_eq!(png.bytes.as_ptr(), original);

    for (bytes, mime) in [
        (JPEG, "image/jpeg"),
        (GIF, "image/gif"),
        (WEBP, "image/webp"),
    ] {
        let mut media = media(bytes);
        prepare(&mut media, 16, LIMIT, ImageResizeFallback::Passthrough)?;
        assert_eq!(media.mime.as_deref(), Some(mime));
        assert_eq!(media.bytes.as_ref(), bytes);
        for end in [0, 7, 8, bytes.len() - 1] {
            assert!(inspect(&bytes[..end], LIMIT).is_err());
        }
    }
    assert_exec_error(b"", LIMIT, "truncated");
    assert_exec_error(&PNG[..7], LIMIT, "truncated");
    assert_exec_error(b"not an image", LIMIT, "unknown image format");

    let mut gif = GIF.to_vec();
    gif[6..8].fill(0);
    assert_exec_error(&gif, LIMIT, "image dimensions");
    let mut jpeg = JPEG.to_vec();
    let sof = jpeg
        .windows(2)
        .position(|pair| pair == b"\xff\xc0")
        .unwrap();
    jpeg[sof + 5..sof + 7].fill(0);
    assert_exec_error(&jpeg, LIMIT, "image dimensions");
    assert_exec_error(WEBP, 5, "image dimensions");
    Ok(())
}

#[test]
fn rejects_corrupt_png_structure_and_crc() {
    let mut bad_length = PNG_SIGNATURE.to_vec();
    append_png_chunk(&mut bad_length, b"IHDR", &[0; 12]);
    assert_exec_error(&bad_length, LIMIT, "IHDR length");

    let mut bad_crc = PNG.to_vec();
    bad_crc[32] ^= 1;
    assert_exec_error(&bad_crc, LIMIT, "CRC");

    let mut bad_chunk_length = PNG.to_vec();
    bad_chunk_length[33..37].copy_from_slice(&u32::MAX.to_be_bytes());
    assert_exec_error(&bad_chunk_length, LIMIT, "chunk length");

    let mut bad_idat_crc = PNG.to_vec();
    bad_idat_crc[50] ^= 1;
    assert_exec_error(&bad_idat_crc, LIMIT, "CRC");
    for end in [8, 18, 32, PNG.len() - 1] {
        assert_exec_error(&PNG[..end], LIMIT, "PNG");
    }

    let payload = zlib(&[0, 1]);
    let split_idat = png_with_chunks(
        (1, 1),
        PngColor::Gray,
        8,
        0,
        &[
            (b"IDAT", &payload[..1]),
            (b"tEXt", b"k\0v"),
            (b"IDAT", &payload[1..]),
        ],
    );
    assert_exec_error(&split_idat, LIMIT, "invalid or unsupported PNG chunk");
}

#[test]
fn rejects_dimensions_and_decoded_byte_limits_before_inflating() {
    for (width, height, limit) in [
        (0, 10, LIMIT),
        (10, 0, LIMIT),
        (8192, 8192, LIMIT),
        (8193, 8192, u64::MAX),
        (65535, 65535, LIMIT),
        (u32::MAX, u32::MAX, u64::MAX),
    ] {
        // Keep IDAT tiny so an attempt to allocate from IHDR would fail this test.
        let bytes = png_with_rows(width, height, PngColor::Gray, &[0, 42]);
        assert_exec_error(&bytes, limit, "image dimensions");
    }
    let rgba = png_with_rows(64, 64, PngColor::Rgba, &[0, 42]);
    assert_exec_error(&rgba, 10000, "decoded pixels exceed max_media_bytes");
}

#[test]
fn rejects_short_long_and_invalid_png_inflation() {
    let short = png_with_rows(1, 1, PngColor::Gray, &[0]);
    assert_exec_error(&short, LIMIT, "shorter than IHDR");
    let long = png_with_rows(1, 1, PngColor::Gray, &[0, 1, 2]);
    assert_exec_error(&long, LIMIT, "longer than IHDR");
    let bomb = png_with_rows(1, 1, PngColor::Gray, &vec![0; 1 << 20]);
    assert_exec_error(&bomb, LIMIT, "longer than IHDR");

    let mut checksum = zlib(&[0, 1]);
    *checksum.last_mut().unwrap() ^= 1;
    let corrupt = png_with_chunks((1, 1), PngColor::Gray, 8, 0, &[(b"IDAT", &checksum)]);
    assert_exec_error(&corrupt, LIMIT, "zlib");

    let mut truncated = zlib(&[0; 16]);
    truncated.pop();
    let truncated =
        png_with_chunks((15, 1), PngColor::Gray, 8, 0, &[(b"IDAT", &truncated)]);
    assert_exec_error(&truncated, LIMIT, "zlib");

    let bad_filter = png_with_rows(1, 1, PngColor::Gray, &[5, 1]);
    assert_exec_error(&bad_filter, LIMIT, "scanline filter");
}

#[test]
fn interlace_follows_fallback_and_sixteen_bit_is_an_error() -> Result<()> {
    // Only the first Adam7 pass has data for a 1x1 image.
    let interlaced =
        png_with_chunks((1, 1), PngColor::Gray, 8, 1, &[(b"IDAT", &zlib(&[0, 1]))]);
    let mut passthrough = media(&interlaced);
    prepare(
        &mut passthrough,
        16,
        LIMIT,
        ImageResizeFallback::Passthrough,
    )?;
    assert_eq!(passthrough.bytes.as_ref(), interlaced);
    let error =
        prepare(&mut passthrough, 16, LIMIT, ImageResizeFallback::Error).unwrap_err();
    assert!(error.to_string().contains("interlaced PNG"));

    let sixteen_bit = png_with_chunks(
        (1, 1),
        PngColor::Gray,
        16,
        0,
        &[(b"IDAT", &zlib(&[0, 0, 1]))],
    );
    assert_exec_error(&sixteen_bit, LIMIT, "bit depth must be 8");
    Ok(())
}

#[test]
fn reverses_all_filters_across_multiple_idat_chunks() -> Result<()> {
    let expected = [
        [1, 218, 2, 31, 53, 68],
        [2, 199, 47, 0, 40, 155],
        [23, 128, 129, 42, 0, 17],
        [31, 89, 1, 60, 7, 155],
        [255, 178, 5, 126, 250, 201],
    ];
    let filtered = [
        [0, 1, 218, 2, 31, 53, 68],
        [1, 2, 199, 47, 254, 97, 108],
        [2, 21, 185, 82, 42, 216, 118],
        [3, 20, 25, 193, 24, 219, 146],
        [4, 224, 89, 4, 127, 161, 46],
    ];
    let payload = zlib(filtered.as_flattened());
    let bytes = png_with_chunks(
        (2, 5),
        PngColor::Rgb,
        8,
        0,
        &[
            (b"IDAT", &[]),
            (b"IDAT", &payload[..1]),
            (b"IDAT", &payload[1..]),
        ],
    );
    let (dimensions, color, pixels) = decode(&bytes)?;
    assert_eq!((dimensions, color), ((2, 5), PngColor::Rgb));
    assert_eq!(pixels, expected);
    Ok(())
}

#[test]
fn resizes_and_round_trips_area_averaged_pixels_and_alpha() -> Result<()> {
    let mut rgb = media(PNG);
    prepare(&mut rgb, 16, LIMIT, ImageResizeFallback::Error)?;
    let (dimensions, color, pixels) = decode(&rgb.bytes)?;
    assert_eq!((dimensions, color), ((16, 8), PngColor::Rgb));
    assert_eq!(&pixels[0][..3], &[2, 4, 1]);
    let mut same = media(PNG);
    prepare(&mut same, 16, LIMIT, ImageResizeFallback::Error)?;
    assert_eq!(rgb.bytes, same.bytes);

    for (color, line, expected) in [
        (PngColor::Gray, &[0, 100, 220][..], &[160][..]),
        (PngColor::GrayAlpha, &[0, 100, 0, 220, 255], &[220, 128]),
        (
            PngColor::Rgba,
            &[0, 255, 0, 0, 0, 40, 60, 80, 128],
            &[40, 60, 80, 64],
        ),
    ] {
        let bytes = png_with_rows(2, 32, color, &line.repeat(32));
        let mut media = media(&bytes);
        prepare(&mut media, 16, LIMIT, ImageResizeFallback::Error)?;
        let (dimensions, actual_color, pixels) = decode(&media.bytes)?;
        assert_eq!((dimensions, actual_color), ((1, 16), color));
        assert!(pixels.iter().all(|pixel| pixel == expected));
    }

    let transparent = png_with_chunks(
        (2, 32),
        PngColor::Gray,
        8,
        0,
        &[
            (b"tRNS", &[0, 100]),
            (b"IDAT", &zlib(&[0, 100, 220].repeat(32))),
        ],
    );
    let mut media = media(&transparent);
    prepare(&mut media, 16, LIMIT, ImageResizeFallback::Error)?;
    let (dimensions, color, pixels) = decode(&media.bytes)?;
    assert_eq!((dimensions, color), ((1, 16), PngColor::GrayAlpha));
    assert!(pixels.iter().all(|pixel| pixel == &[220, 128]));
    Ok(())
}

#[test]
fn large_landscape_and_small_images_keep_aspect_ratio() -> Result<()> {
    let mut scanline = vec![133; 4001];
    scanline[0] = 0;
    let large = png_with_rows(4000, 3000, PngColor::Gray, &scanline.repeat(3000));
    let mut resized = media(&large);
    prepare(&mut resized, 1024, LIMIT, ImageResizeFallback::Error)?;
    let (dimensions, color, pixels) = decode(&resized.bytes)?;
    assert_eq!((dimensions, color), ((1024, 768), PngColor::Gray));
    assert!(pixels.iter().flatten().all(|gray| *gray == 133));

    for (width, height) in [(1000, 500), (1, 1)] {
        let bytes = png_with_rows(
            width,
            height,
            PngColor::Gray,
            &vec![0; ((width + 1) * height) as usize],
        );
        let mut small = media(&bytes);
        let original = small.bytes.as_ptr();
        prepare(&mut small, 1024, LIMIT, ImageResizeFallback::Error)?;
        assert_eq!(small.bytes.as_ref(), bytes);
        assert_eq!(small.bytes.as_ptr(), original);
    }

    let narrow = png_with_rows(1, 128, PngColor::Gray, &[0, 100].repeat(128));
    let mut resized = media(&narrow);
    prepare(&mut resized, 16, LIMIT, ImageResizeFallback::Error)?;
    assert_eq!(decode(&resized.bytes)?.0, (1, 16));
    Ok(())
}

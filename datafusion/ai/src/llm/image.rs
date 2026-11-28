// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::borrow::Cow;
use std::io::{Read, Take, Write};

use datafusion_common::{Result, exec_datafusion_err, exec_err};
use flate2::{Compression, bufread::ZlibDecoder, write::ZlibEncoder};

use super::options::ImageResizeFallback;
use crate::ResolvedMedia;

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const PNG_MIME: &str = "image/png";
const MAX_PIXELS: u64 = 64 << 20;

#[derive(Debug)]
enum ImageDecision<'a> {
    Resize(Png<'a>),
    Passthrough {
        mime: &'static str,
        format: &'static str,
    },
}

pub(super) fn prepare(
    media: &mut ResolvedMedia,
    max_edge: u32,
    max_media_bytes: u64,
    fallback: ImageResizeFallback,
) -> Result<()> {
    let mime = match inspect(&media.bytes, max_media_bytes)? {
        ImageDecision::Resize(png) => {
            if let Some(bytes) = png.resize(max_edge as usize)? {
                media.bytes = bytes.into();
                media.mime = Some(PNG_MIME.into());
                return Ok(());
            }
            PNG_MIME
        }
        ImageDecision::Passthrough { mime, format } => {
            if fallback == ImageResizeFallback::Error {
                return exec_err!("image resize is unsupported for {format}");
            }
            mime
        }
    };
    media.mime.get_or_insert_with(|| mime.into());
    Ok(())
}

fn inspect(bytes: &[u8], max_media_bytes: u64) -> Result<ImageDecision<'_>> {
    if bytes.len() < 8 {
        return exec_err!("image header is truncated");
    }
    if let Some(chunks) = bytes.strip_prefix(PNG_SIGNATURE) {
        let png = Png::parse(chunks, max_media_bytes)?;
        return Ok(if png.interlaced {
            ImageDecision::Passthrough {
                mime: PNG_MIME,
                format: "interlaced PNG",
            }
        } else {
            ImageDecision::Resize(png)
        });
    }
    let (format, mime, dimensions) = if bytes.starts_with(b"\xff\xd8\xff") {
        ("JPEG", "image/jpeg", jpeg_dimensions(bytes))
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        ("GIF", "image/gif", gif_dimensions(bytes))
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP") {
        ("WebP", "image/webp", webp_dimensions(bytes))
    } else {
        return exec_err!("unknown image format");
    };
    let (width, height) =
        dimensions.ok_or_else(|| exec_datafusion_err!("invalid {format} header"))?;
    checked_pixels(width, height, max_media_bytes)?;
    Ok(ImageDecision::Passthrough { mime, format })
}

fn checked_pixels(width: u32, height: u32, max_media_bytes: u64) -> Result<u64> {
    let limit = MAX_PIXELS.min(max_media_bytes);
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(|| exec_datafusion_err!("image pixel count overflow"))?;
    if pixels == 0 || pixels > limit {
        return exec_err!("image dimensions must declare between 1 and {limit} pixels");
    }
    Ok(pixels)
}

fn jpeg_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let mut segments = bytes.strip_prefix(b"\xff\xd8")?;
    let mut dimensions = None;
    while let Some(rest) = segments.strip_prefix(b"\xff") {
        let start = rest.iter().position(|&byte| byte != 0xff)?;
        let (&marker, rest) = rest[start..].split_first()?;
        if marker == 0x01 {
            segments = rest;
            continue;
        }
        if matches!(marker, 0x00 | 0xd0..=0xd9) {
            return None;
        }
        let length = usize::from(u16::from_be_bytes(*rest.first_chunk::<2>()?));
        let header = rest.get(2..length)?;
        segments = &rest[length..];
        match marker {
            0xc0..=0xc3 | 0xc5..=0xc7 | 0xc9..=0xcb | 0xcd..=0xcf => {
                let [_, h1, h2, w1, w2, components, entries @ ..] = header else {
                    return None;
                };
                if *components == 0 || entries.len() != usize::from(*components) * 3 {
                    return None;
                }
                dimensions = Some((
                    u32::from(u16::from_be_bytes([*w1, *w2])),
                    u32::from(u16::from_be_bytes([*h1, *h2])),
                ));
            }
            0xda => {
                let (&components, rest) = header.split_first()?;
                return (components != 0
                    && rest.len() == usize::from(components) * 2 + 3
                    && segments.windows(2).rev().any(|pair| pair == b"\xff\xd9"))
                .then_some(dimensions)
                .flatten();
            }
            _ => {}
        }
    }
    None
}

fn gif_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let &[w1, w2, h1, h2, flags, _, _] = bytes.get(6..13)? else {
        return None;
    };
    let color_table_size = if flags & 0x80 != 0 {
        3 << ((flags & 7) + 1)
    } else {
        0
    };
    (bytes.len() > 13 + color_table_size && bytes.last() == Some(&0x3b)).then_some((
        u32::from(u16::from_le_bytes([w1, w2])),
        u32::from(u16::from_le_bytes([h1, h2])),
    ))
}

fn webp_dimensions(bytes: &[u8]) -> Option<(u32, u32)> {
    let file_size =
        usize::try_from(u32::from_le_bytes(*bytes.get(4..)?.first_chunk::<4>()?)).ok()?;
    if !file_size.is_multiple_of(2) {
        return None;
    }
    let (name, contents) = bytes
        .get(12..file_size.checked_add(8)?)?
        .split_first_chunk::<4>()?;
    let (length, contents) = contents.split_first_chunk::<4>()?;
    let length = usize::try_from(u32::from_le_bytes(*length)).ok()?;
    let (contents, rest) = contents.split_at_checked(length)?;
    if !length.is_multiple_of(2) && rest.first() != Some(&0) {
        return None;
    }
    match (name, contents) {
        (b"VP8 ", [tag, _, _, 0x9d, 0x01, 0x2a, w1, w2, h1, h2, ..]) if tag & 1 == 0 => {
            Some((
                u32::from(u16::from_le_bytes([*w1, *w2]) & 0x3fff),
                u32::from(u16::from_le_bytes([*h1, *h2]) & 0x3fff),
            ))
        }
        (b"VP8L", [0x2f, b1, b2, b3, b4, ..]) => {
            let packed = u32::from_le_bytes([*b1, *b2, *b3, *b4]);
            (packed >> 29 == 0)
                .then_some(((packed & 0x3fff) + 1, ((packed >> 14) & 0x3fff) + 1))
        }
        (b"VP8X", [_, _, _, _, w1, w2, w3, h1, h2, h3]) => Some((
            u32::from_le_bytes([*w1, *w2, *w3, 0]) + 1,
            u32::from_le_bytes([*h1, *h2, *h3, 0]) + 1,
        )),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PngColor {
    Gray = 0,
    Rgb = 2,
    GrayAlpha = 4,
    Rgba = 6,
}

impl PngColor {
    fn color_channels(self) -> usize {
        match self {
            Self::Gray | Self::GrayAlpha => 1,
            Self::Rgb | Self::Rgba => 3,
        }
    }

    fn has_alpha(self) -> bool {
        matches!(self, Self::GrayAlpha | Self::Rgba)
    }

    fn channels(self) -> usize {
        self.color_channels() + usize::from(self.has_alpha())
    }

    fn with_alpha(self) -> Self {
        match self {
            Self::Gray | Self::GrayAlpha => Self::GrayAlpha,
            Self::Rgb | Self::Rgba => Self::Rgba,
        }
    }
}

#[derive(Debug)]
struct Png<'a> {
    width: usize,
    height: usize,
    color: PngColor,
    interlaced: bool,
    inflated_len: u64,
    transparency: Option<[u8; 3]>,
    idat: Cow<'a, [u8]>,
}

impl<'a> Png<'a> {
    fn parse(bytes: &'a [u8], max_media_bytes: u64) -> Result<Self> {
        let Some((b"IHDR", header, mut chunks)) = png_chunk(bytes) else {
            return exec_err!("invalid PNG IHDR chunk or CRC");
        };
        let Some((
            &[w1, w2, w3, w4, h1, h2, h3, h4],
            &[depth, color, compression, filter, interlace],
        )) = header.split_first_chunk::<8>()
        else {
            return exec_err!("invalid PNG IHDR length");
        };
        let width = u32::from_be_bytes([w1, w2, w3, w4]);
        let height = u32::from_be_bytes([h1, h2, h3, h4]);
        let pixels = checked_pixels(width, height, max_media_bytes)?;
        if depth != 8 {
            return exec_err!("PNG bit depth must be 8");
        }
        let color = match color {
            0 => PngColor::Gray,
            2 => PngColor::Rgb,
            4 => PngColor::GrayAlpha,
            6 => PngColor::Rgba,
            _ => return exec_err!("unsupported PNG color type"),
        };
        if compression != 0 || filter != 0 || interlace > 1 {
            return exec_err!("invalid PNG compression, filter, or interlace method");
        }
        let pixel_bytes = pixels
            .checked_mul(color.channels() as u64)
            .ok_or_else(|| exec_datafusion_err!("PNG decoded size overflow"))?;
        if pixel_bytes > max_media_bytes {
            return exec_err!(
                "PNG decoded pixels exceed max_media_bytes {max_media_bytes}"
            );
        }
        // One filter byte precedes each non-interlaced scanline.
        let inflated_len = pixel_bytes + u64::from(height);
        let mut idat = Cow::Borrowed(&[][..]);
        let mut seen_idat = false;
        let mut ended_idat = false;
        let mut seen_palette = false;
        let mut transparency = None;
        loop {
            let Some((kind, data, rest)) = png_chunk(chunks) else {
                return exec_err!("invalid PNG chunk length or CRC");
            };
            chunks = rest;
            match kind {
                b"IDAT" if !ended_idat => {
                    seen_idat = true;
                    if idat.is_empty() {
                        idat = Cow::Borrowed(data);
                    } else {
                        idat.to_mut().extend_from_slice(data);
                    }
                }
                b"IEND" if data.is_empty() && rest.is_empty() && idat.len() >= 8 => {
                    return Ok(Self {
                        width: width as usize,
                        height: height as usize,
                        color,
                        interlaced: interlace == 1,
                        inflated_len,
                        transparency,
                        idat,
                    });
                }
                b"PLTE"
                    if !seen_palette
                        && !seen_idat
                        && transparency.is_none()
                        && color.color_channels() == 3
                        && (3..=768).contains(&data.len())
                        && data.len().is_multiple_of(3) =>
                {
                    seen_palette = true;
                }
                b"tRNS" if !seen_idat && transparency.is_none() && !color.has_alpha() => {
                    transparency = Some(match (color, data) {
                        (PngColor::Gray, &[0, gray]) => [gray, 0, 0],
                        (PngColor::Rgb, &[0, red, 0, green, 0, blue]) => {
                            [red, green, blue]
                        }
                        _ => return exec_err!("invalid PNG transparency chunk"),
                    });
                }
                b"tRNS" => return exec_err!("invalid PNG transparency chunk order"),
                _ if kind[0].is_ascii_lowercase()
                    && kind.iter().all(u8::is_ascii_alphabetic) =>
                {
                    ended_idat = seen_idat;
                }
                _ => return exec_err!("invalid or unsupported PNG chunk"),
            }
        }
    }

    fn rows(&self) -> PngRows<'_> {
        let channels = self.color.channels();
        let stride = self.width * channels + 1;
        PngRows {
            decoder: ZlibDecoder::new(self.idat.as_ref()).take(self.inflated_len),
            current: vec![0; stride],
            previous: vec![0; stride],
            channels,
        }
    }

    fn resize(self, max_edge: usize) -> Result<Option<Vec<u8>>> {
        let mut rows = self.rows();
        let long_edge = self.width.max(self.height);
        if long_edge <= max_edge {
            for _ in 0..self.height {
                rows.read(false)?;
            }
            rows.finish()?;
            return Ok(None);
        }

        let scale = |edge: usize| {
            let long_edge = long_edge as u64;
            ((edge as u64 * max_edge as u64 + long_edge / 2) / long_edge).max(1) as usize
        };
        let (width, height) = (scale(self.width), scale(self.height));
        let output = self.resample(&mut rows, width, height)?;
        rows.finish()?;
        Ok(Some(output))
    }

    fn resample(
        &self,
        rows: &mut PngRows<'_>,
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>> {
        let output_color = if self.transparency.is_some() {
            self.color.with_alpha()
        } else {
            self.color
        };
        let color_channels = self.color.color_channels();
        let channels = self.color.channels();
        let output_channels = output_color.channels();
        let mut png = PNG_SIGNATURE.to_vec();
        let mut header = [0; 13];
        header[..4].copy_from_slice(&(width as u32).to_be_bytes());
        header[4..8].copy_from_slice(&(height as u32).to_be_bytes());
        header[8] = 8;
        header[9] = output_color as u8;
        append_png_chunk(&mut png, b"IHDR", &header);

        // Write the compressed payload into the output buffer and fill its length afterwards.
        let idat_start = png.len();
        png.extend_from_slice(b"\0\0\0\0IDAT");
        let mut encoder = ZlibEncoder::new(png, Compression::new(6));
        let edges: Vec<_> = (0..=width)
            .map(|x| scale_floor(x, self.width, width) * channels)
            .collect();
        let mut sums = vec![[0u64; 4]; width];
        let mut row = vec![0; width * output_channels + 1];
        let mut top = 0;

        for y in 0..height {
            let bottom = scale_floor(y + 1, self.height, height);
            sums.fill([0; 4]);
            for _ in top..bottom {
                let src = rows.read(true)?;
                for (sum, bounds) in sums.iter_mut().zip(edges.windows(2)) {
                    for pixel in src[bounds[0]..bounds[1]].chunks_exact(channels) {
                        let alpha = if self.color.has_alpha() {
                            pixel[color_channels]
                        } else if self
                            .transparency
                            .is_some_and(|key| pixel == &key[..color_channels])
                        {
                            0
                        } else {
                            255
                        };
                        let alpha = u64::from(alpha);
                        for (sum, value) in sum.iter_mut().take(color_channels).zip(pixel)
                        {
                            *sum += u64::from(*value) * alpha;
                        }
                        sum[3] += alpha;
                    }
                }
            }
            for ((sum, bounds), pixel) in sums
                .iter()
                .zip(edges.windows(2))
                .zip(row[1..].chunks_exact_mut(output_channels))
            {
                let count =
                    ((bottom - top) * ((bounds[1] - bounds[0]) / channels)) as u64;
                for (value, sum_value) in pixel.iter_mut().take(color_channels).zip(sum) {
                    *value = rounded_average(*sum_value, sum[3].max(1));
                }
                if output_color.has_alpha() {
                    pixel[color_channels] = rounded_average(sum[3], count);
                }
            }
            encoder.write_all(&row)?;
            top = bottom;
        }
        let mut png = encoder.finish()?;
        let length = u32::try_from(png.len() - idat_start - 8)
            .map_err(|_| exec_datafusion_err!("encoded PNG IDAT size overflow"))?;
        png[idat_start..idat_start + 4].copy_from_slice(&length.to_be_bytes());
        let crc = crc32(&png[idat_start + 4..]);
        png.extend_from_slice(&crc.to_be_bytes());
        append_png_chunk(&mut png, b"IEND", &[]);
        Ok(png)
    }
}

fn scale_floor(edge: usize, source: usize, target: usize) -> usize {
    (edge as u64 * source as u64 / target as u64) as usize
}

fn rounded_average(sum: u64, count: u64) -> u8 {
    ((sum + count / 2) / count) as u8
}

struct PngRows<'a> {
    decoder: Take<ZlibDecoder<&'a [u8]>>,
    current: Vec<u8>,
    previous: Vec<u8>,
    channels: usize,
}

impl PngRows<'_> {
    fn read(&mut self, unfilter: bool) -> Result<&[u8]> {
        self.decoder.read_exact(&mut self.current).map_err(|_| {
            exec_datafusion_err!(
                "invalid PNG zlib stream or shorter than IHDR scanline length"
            )
        })?;
        let filter = self.current[0];
        if filter > 4 {
            return exec_err!("invalid PNG scanline filter");
        }
        if !unfilter {
            return Ok(&self.current[1..]);
        }
        if filter != 0 {
            let current = &mut self.current[1..];
            let previous = &self.previous[1..];
            for (i, &above) in previous.iter().enumerate() {
                let (left, corner) = i
                    .checked_sub(self.channels)
                    .map_or((0, 0), |j| (current[j], previous[j]));
                let predictor = match filter {
                    1 => left,
                    2 => above,
                    3 => u8::midpoint(left, above),
                    _ => paeth(left, above, corner),
                };
                current[i] = current[i].wrapping_add(predictor);
            }
        }
        std::mem::swap(&mut self.current, &mut self.previous);
        Ok(&self.previous[1..])
    }

    fn finish(self) -> Result<()> {
        // Read at most one extra byte to detect excess data and verify the zlib trailer.
        let extra = self
            .decoder
            .into_inner()
            .read(&mut [0])
            .map_err(|_| exec_datafusion_err!("invalid PNG zlib trailer"))?;
        if extra != 0 {
            return exec_err!("PNG zlib stream is longer than IHDR scanline length");
        }
        Ok(())
    }
}

fn paeth(left: u8, above: u8, corner: u8) -> u8 {
    let (a, b, c) = (i16::from(left), i16::from(above), i16::from(corner));
    let (da, db, dc) = ((b - c).abs(), (a - c).abs(), (a + b - 2 * c).abs());
    if da <= db && da <= dc {
        left
    } else if db <= dc {
        above
    } else {
        corner
    }
}

fn png_chunk(bytes: &[u8]) -> Option<(&[u8; 4], &[u8], &[u8])> {
    let (length, bytes) = bytes.split_first_chunk::<4>()?;
    let (kind, contents) = bytes.split_first_chunk::<4>()?;
    let length = usize::try_from(u32::from_be_bytes(*length)).ok()?;
    let (data, trailer) = contents.split_at_checked(length)?;
    let (crc, rest) = trailer.split_first_chunk::<4>()?;
    (crc32(&bytes[..4 + length]) == u32::from_be_bytes(*crc))
        .then_some((kind, data, rest))
}

fn append_png_chunk(output: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = output.len();
    output.extend_from_slice(kind);
    output.extend_from_slice(data);
    let crc = crc32(&output[start..]);
    output.extend_from_slice(&crc.to_be_bytes());
}

fn crc32(data: &[u8]) -> u32 {
    const TABLE: [u32; 256] = {
        let mut table = [0; 256];
        let mut index = 0;
        while index < table.len() {
            let mut value = index as u32;
            let mut bit = 0;
            while bit < 8 {
                value = (value >> 1) ^ (0xedb8_8320 & (value & 1).wrapping_neg());
                bit += 1;
            }
            table[index] = value;
            index += 1;
        }
        table
    };

    !data.iter().fold(u32::MAX, |crc, &byte| {
        (crc >> 8) ^ TABLE[usize::from(crc as u8 ^ byte)]
    })
}

#[cfg(test)]
mod tests;

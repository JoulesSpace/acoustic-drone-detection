# syntax=docker/dockerfile:1
#
# Multi-stage build for the `drone` host CLI. The DSP/detection crates are
# no_std-friendly and could later be cross-built for esp32/riscv firmware; this
# image is the *host* analysis/test runtime, not the firmware image.

FROM rust:1.92-slim-bookworm AS builder
WORKDIR /build

# Copy all sibling crates (no workspace yet - path deps resolve relative to
# drone-cli). Copying the whole tree keeps the path deps valid.
COPY crates ./crates

WORKDIR /build/crates/drone-cli
# Cache the cargo registry and the target dir across builds. Because cache
# mounts are not persisted into the image layer, copy the binary out within the
# same RUN so the runtime stage can pick it up.
RUN --mount=type=cache,target=/usr/local/cargo/registry \
    --mount=type=cache,target=/build/crates/drone-cli/target \
    cargo build --release && \
    cp target/release/drone /usr/local/bin/drone

FROM debian:bookworm-slim AS runtime
RUN useradd --create-home app
COPY --from=builder /usr/local/bin/drone /usr/local/bin/drone
USER app
WORKDIR /data
ENTRYPOINT ["drone"]
CMD ["--help"]

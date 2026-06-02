# Convenience wrappers over docker compose (Docker Desktop must be running).
# These mirror the documented commands in the README so you do not have to
# remember the full `docker compose run` invocations.

.PHONY: help build check bench plot data figures synth clean

help:
	@echo "build    build the runtime detector image"
	@echo "check    fmt --check + clippy -D warnings + tests + no_std/riscv builds"
	@echo "data     stream a balanced DADS subset into ./data/dads"
	@echo "bench    run the detection benchmark (synthetic by default)"
	@echo "plot     render benchmark plots from results JSON"
	@echo "figures  regenerate the signal-chain infographic"
	@echo "synth    synth a test drone signal and analyze it"

build:
	docker compose build detector

check:
	docker compose run --rm dev

data:
	docker compose run --rm data --per-class 300

bench:
	docker compose run --rm bench

plot:
	docker compose run --rm plot

figures:
	docker compose run --rm --entrypoint python plot scripts/infographic.py

synth:
	docker compose run --rm detector synth   --out /data/test.wav --fundamental 120
	docker compose run --rm detector analyze --input /data/test.wav

clean:
	docker compose down -v

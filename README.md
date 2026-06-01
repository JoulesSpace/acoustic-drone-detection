# Acoustic-Drone-Detection

Detecting drones accoustically.

## Scope

Design an acoustic drone detection pipeline, and validate your ideas in simulation.

Questions we answer with this porject:

- Detection. What makes a drone signature distinguishable from background sound? What features or representations would you feed a model, and how would you know it's actually working?
- Direction of Arrival. If you used multiple microphones, how would you estimate where the drone is? What does array geometry buy you, and what are the trade-offs?
- Robustness. Real deployments are noisy — literally. Wind, rain, overlapping sources, varying drone types. How would you stress-test your approach? This is where simulation earns its keep: you control what you throw at it.
- System Design. What would a real deployment look like? How many microphones, in what configuration, at what sample rate? What detection range would you expect and why? What are the fundamental physical limits?

## Contributing

Welcome! fork -> branch `[name]/feat|fix-[feat/fix-name]` -> pr -> fix feedback -> get merged

## License

> what is the license that makes people need to open source if they modify or use it?

Google says AGPLv3.

let license = "AGPLv3";

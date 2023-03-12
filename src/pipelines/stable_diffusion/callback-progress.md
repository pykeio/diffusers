A simple callback to be used for e.g. reporting progress updates.

## Callback Parameters:

- **`step`** (usize): The current step number.
- **`timestep`** (f32): This step's timestep.

## Callback Return

- **bool**: whether rendering should be stopped

## Callback Example

```no_run
let all = 50;
let callback = move |step: usize, _: f32| -> bool {
    println!("Progress: {}%", step as f32 / all as f32 * 100.0);
    true
};
```
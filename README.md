## Burn issues discoverd while working with the multipa model
### Running
The model is exported from pytorch to onnx and imported to Rust via burn-onnx<br>
To export the model run<br>
`uv run export.py`<br>
This exports the model, and runs onnxsim on it, to dodge the [weight tensor must be present](https://github.com/tracel-ai/burn-onnx/issues/346) issue<br>
The rust code can be then run with standard `cargo run -r`
use std::time::Instant;

use burn::Tensor;

mod model {
    include!(concat!(env!("OUT_DIR"), "/model/multipa_sim.rs"));
}

// Easy backend swap comment/uncomment

use burn::backend::{flex::FlexDevice, Flex};
type Device = FlexDevice;
type Backend = Flex;

// use burn::backend::{cpu::CpuDevice, Cpu};
// type Device = CpuDevice;
// type Backend = Cpu;

fn main() {
    let device: Device = FlexDevice;
    // let device: Device = CpuDevice;
    let model: model::Model<Backend> = model::Model::default();

    let input_tensor = Tensor::zeros([1, 16_000 * 10], &device);

    let now = Instant::now();

    let output = model.forward(input_tensor);

    let elapsed = now.elapsed();
    println!("Inference took {:.2?}", elapsed);

    println!("Model output: {:?}", output);
}

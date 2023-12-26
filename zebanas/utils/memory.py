import subprocess


def set_clock_speed(device, clock_speed):
    process = subprocess.Popen(
        "nvidia-smi",
        stdout=subprocess.PIPE,
        shell=True
    )
    stdout, _ = process.communicate()
    process = subprocess.run(
        f"sudo nvidia-smi -pm ENABLED -i {device}",
        shell=True
    )
    process = subprocess.run(
        f"sudo nvidia-smi -lgc {clock_speed} -i {device}",
        shell=True
    )


def reset_clock_speed(device):
    subprocess.run(f"sudo nvidia-smi -pm ENABLED -i {device}", shell=True)
    subprocess.run(f"sudo nvidia-smi -rgc -i {device}", shell=True)


def flush_cache(model, xs):
    for x in xs:
        x.zero_()
    [p.data.zero_() for p in model.parameters()]

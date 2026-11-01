import yaml, wandb
for f in ["sweep_optimizer.yaml", "sweep_activation.yaml",
          "sweep_lr.yaml", "sweep_batch.yaml", "sweep_epochs.yaml"]:
    cfg = yaml.safe_load(open(f))
    cfg["name"] = f.replace("sweep_", "").replace(".yaml", "")   # sweep name shown in W&B
    sweep_id = wandb.sweep(cfg, project="swaminathan_vgg6_CIFAR-10")
    print(f"\n===== {cfg['name']}: {sweep_id} =====\n")
    wandb agent --project swaminathan_vgg6_CIFAR-10 {sweep_id}
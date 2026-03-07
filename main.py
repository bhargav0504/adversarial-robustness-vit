#!/usr/bin/env python3
"""
Adversarial Robustness Framework — Main Entry Point
UTA AI Course Project

Commands:
  python main.py train                  Fine-tune ViT + ResNet on CIFAR-10
  python main.py train --model vit      Fine-tune ViT only
  python main.py train --adv-train      Fine-tune + adversarial training
  python main.py evaluate               Benchmark clean + robust accuracy
  python main.py demo                   Launch interactive Gradio demo

Typical workflow:
  1. python main.py train               (run once, saves checkpoints/)
  2. python main.py train --adv-train   (optional, adds adv-trained checkpoints)
  3. python main.py evaluate            (generates results/ table + plot)
  4. python main.py demo                (open http://localhost:7860)
"""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(
        description='Adversarial Robustness Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        'command', choices=['train', 'evaluate', 'demo'],
        help='Action to run'
    )
    parser.add_argument('--model', choices=['vit', 'resnet', 'both'], default='both')
    parser.add_argument('--adv-train', action='store_true',
                        help='Run adversarial training after standard fine-tuning')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--device', default='auto',
                        help='cpu | cuda | auto  (default: auto)')
    args = parser.parse_args()

    py = sys.executable

    if args.command == 'train':
        cmd = [py, 'train.py',
               '--config', args.config,
               '--model',  args.model,
               '--device', args.device]
        if args.adv_train:
            cmd.append('--adv-train')
        subprocess.run(cmd, check=True)

    elif args.command == 'evaluate':
        subprocess.run(
            [py, 'evaluate.py', '--config', args.config, '--device', args.device],
            check=True
        )

    elif args.command == 'demo':
        subprocess.run([py, 'demo/app.py'], check=True)


if __name__ == '__main__':
    main()

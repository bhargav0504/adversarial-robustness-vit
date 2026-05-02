import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Adversarial Robustness Framework')
    parser.add_argument('command', choices=['train', 'evaluate', 'demo'])
    parser.add_argument('--model', choices=['vit', 'resnet', 'both'], default='both')
    parser.add_argument('--adv-train', action='store_true')
    parser.add_argument('--config', default='configs/config.yaml')
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()

    py = sys.executable

    if args.command == 'train':
        cmd = [py, 'train.py', '--config', args.config, '--model', args.model, '--device', args.device]
        if args.adv_train:
            cmd.append('--adv-train')
        subprocess.run(cmd, check=True)

    elif args.command == 'evaluate':
        subprocess.run([py, 'evaluate.py', '--config', args.config, '--device', args.device], check=True)

    elif args.command == 'demo':
        subprocess.run([py, 'demo/app.py'], check=True)


if __name__ == '__main__':
    main()

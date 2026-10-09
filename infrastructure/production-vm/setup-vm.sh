#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 22.04/24.04 VM for infrastructure/production-vm/compose.yml.
#   sudo bash setup-vm.sh [swap-size, default 4G]
# Installs Docker Engine + Compose plugin (official repository), adds swap (RAG Core peaks at about
# 1.1 GB RAM while ingesting large files; swap keeps a 2 GB VM from out-of-memory kills) and allows
# only SSH, HTTP and HTTPS through the host firewall. The cloud firewall/security group must match.
set -euo pipefail
SWAP_SIZE="${1:-4G}"

if ! command -v docker >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y ca-certificates curl gnupg
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  . /etc/os-release
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -y
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker
[ -n "${SUDO_USER:-}" ] && usermod -aG docker "$SUDO_USER"

if ! swapon --show | grep -q /swapfile; then
  fallocate -l "$SWAP_SIZE" /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
  sysctl -w vm.swappiness=10
  echo 'vm.swappiness=10' > /etc/sysctl.d/99-advancerag.conf
fi

if command -v ufw >/dev/null 2>&1; then
  ufw allow OpenSSH
  ufw allow 80/tcp
  ufw allow 443/tcp
  ufw allow 443/udp
  ufw --force enable
fi

docker --version
docker compose version
free -h
echo "Done. Log out and back in (docker group), then follow infrastructure/production-vm/README.md."

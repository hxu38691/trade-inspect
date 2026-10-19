# Refresh paper fills from Alibaba ECS

On the box (has `gh` auth as hxu38691):

```bash
PASS_FILE=... # do not commit password
sshpass -f "$PASS_FILE" ssh -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no root@47.89.164.178 \
  'tar czf - -C /root/ibkr/paper_sync/out .' | tar xzf - -C /workspace/trade-inspect/inspect/paper_fills
cd /workspace/trade-inspect && git add inspect/paper_fills index.html && git commit -m "Update IB paper fills dashboard" && git push origin main
```

ECS continuous sync: `systemctl status paper-fill-sync` (every ~2 min, read-only).
Gateway autostart: `systemctl is-enabled ibgateway-paper` (enabled; current process may be pre-systemd until reboot).

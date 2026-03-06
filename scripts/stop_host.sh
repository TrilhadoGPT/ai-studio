#!/usr/bin/env bash
set -euo pipefail

pkill -f 'api.image_api:app' >/dev/null 2>&1 || true
pkill -f 'api.video_api:app' >/dev/null 2>&1 || true
echo "🛑 APIs interrompidas (image/video)."

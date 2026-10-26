#!/bin/sh
# Install a downloaded LBMSim release next to the running binary.
#
# Usage: update.sh <app-dir> <archive.tar.gz>
#
# Called by UpdateManager::ApplyUpdate() after the archive's SHA-256 has been
# verified. The binary is replaced with rename(), which is safe while the old
# binary is still running; the caller then execv()s the new one.
set -u

APP_DIR=${1:?usage: update.sh <app-dir> <archive>}
ARCHIVE=${2:?usage: update.sh <app-dir> <archive>}
LOCK_FILE=/tmp/lbm_update.lock

if [ -e "$LOCK_FILE" ]; then
    echo "Update already in progress ($LOCK_FILE exists)" >&2
    exit 1
fi
touch "$LOCK_FILE"
TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"; rm -f "$LOCK_FILE"' EXIT

if ! tar -xzf "$ARCHIVE" -C "$TMP_DIR"; then
    echo "Cannot unpack $ARCHIVE" >&2
    exit 1
fi

if [ ! -f "$TMP_DIR/executables" ] || [ ! -f "$TMP_DIR/version.info" ]; then
    echo "Update package is incomplete (expected executables and version.info)" >&2
    exit 1
fi

# Copy next to the target first, then rename over it: atomic on the same filesystem.
cp "$TMP_DIR/executables" "$APP_DIR/executables.new" &&
    chmod +x "$APP_DIR/executables.new" &&
    mv -f "$APP_DIR/executables.new" "$APP_DIR/executables" || exit 1

cp "$TMP_DIR/version.info" "$APP_DIR/version.info.new" &&
    mv -f "$APP_DIR/version.info.new" "$APP_DIR/version.info" || exit 1

# Newer packages ship this script too; keep it up to date for the next update.
if [ -f "$TMP_DIR/UpdateLoader/scripts/update.sh" ]; then
    mkdir -p "$APP_DIR/UpdateLoader/scripts"
    cp "$TMP_DIR/UpdateLoader/scripts/update.sh" "$APP_DIR/UpdateLoader/scripts/update.sh.new" &&
        mv -f "$APP_DIR/UpdateLoader/scripts/update.sh.new" "$APP_DIR/UpdateLoader/scripts/update.sh"
fi

rm -f "$ARCHIVE"
echo "Installed version: $(cat "$APP_DIR/version.info")"

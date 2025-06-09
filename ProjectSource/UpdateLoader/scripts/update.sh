# Get absolute path of executable
APP_DIR=$(dirname "$(readlink -f "$0")")/../..
LOCK_FILE="/tmp/lbm_update.lock"

# Prevent concurrent updates
if [ -f "$LOCK_FILE" ]; then
    echo "Update already in progress"
    exit 1
fi
touch "$LOCK_FILE"

# Wait for main process to exit
TIMEOUT=10
while pgrep -f "executables" && [ $TIMEOUT -gt 0 ]; do
    sleep 1
    ((TIMEOUT--))
done

if [ $TIMEOUT -eq 0 ]; then
    echo "Failed to terminate running process"
    rm "$LOCK_FILE"
    exit 1
fi

# Extract update to temporary directory
TMP_DIR=$(mktemp -d)
tar -xzf /tmp/lbm_update.tar.gz -C "$TMP_DIR"

# Verify files exist
if [ ! -f "$TMP_DIR/executables" ] || [ ! -f "$TMP_DIR/version.info" ]; then
    echo "Update package corrupted"
    rm -rf "$TMP_DIR" "$LOCK_FILE"
    exit 1
fi

# Atomic file replacement
cp -f "$TMP_DIR/version.info" "$APP_DIR/"
sync
cp -f "$TMP_DIR/executables" "$APP_DIR/"
sync

# Cleanup
rm -rf "$TMP_DIR" /tmp/lbm_update.tar.gz
rm "$LOCK_FILE"

# Restart with new version
echo "Restarting with version: $(cat "$APP_DIR/version.info")"
cd "$APP_DIR"
exec ./executables "$@"
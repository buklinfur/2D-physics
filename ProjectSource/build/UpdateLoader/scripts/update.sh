while pgrep -f "executables"; do
  sleep 1
done

tar -xzf /tmp/lbm_update.tar.gz -C "$(dirname "$0")/../../"
rm /tmp/lbm_update.tar.gz

./executables &
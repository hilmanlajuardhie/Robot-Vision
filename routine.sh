sudo apt upgrade;
sudo apt full-upgrade -y;
sudo snap refresh;
sudo apt autoremove --purge -y;
sudo apt clean;
sudo journalctl --vacuum-time=3d;
rm -rf ~/.cache/thumbnails/*;
rm -rf ~/.local/share/Trash/files/*;
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null;
pip cache purge;
pip check

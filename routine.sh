echo "----- Starting routine.sh -----";
echo "\n----- apt Update && Upgrade -----";
sudo apt upgrade;
sudo apt full-upgrade -y;
echo "\n----- apt autoremove && Clean -----";
sudo apt autoremove --purge -y;
sudo apt clean;
echo "\n----- snap refresh -----";
sudo snap refresh;
#echo "\n----- rm Journcalctl && cache folder -----";
#sudo journalctl --vacuum-time=3d;
#rm -rf ~/.cache/thumbnails/*;
#rm -rf ~/.local/share/Trash/files/*;
#sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches > /dev/null;
echo "\n----- PIP Cache Purge && Check -----";
pip cache purge;
pip check;

echo "\n----- routine.sh Finised ----\n"

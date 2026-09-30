cp ./config/lightpath-camera.service /etc/systemd/system/
systemctl enable lightpath-camera.service

# To remove service, delete the file from /etc/systemd/system/ and run:
# systemctl disable lightpath-camera.service


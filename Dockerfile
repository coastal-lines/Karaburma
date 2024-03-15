FROM hdgigante/python-opencv:4.9.0-debian

RUN apt-get update -y
RUN apt-get install -y python3
RUN apt-get install -y python3-pip

#pyautogui
RUN apt-get install -y dbus-x11


RUN apt-get install -y xvfb
RUN apt install -y python3-xlib python3-tk python3-dev # PyAutoGUI requirements
RUN apt install -y xvfb xserver-xephyr # Monkey-patch PyAutoGUI internals

# Set default folder for application
WORKDIR /Karaburma

# Copy 'requirements.txt' file into application folder
COPY requirements.txt /Karaburma/

# Install specific 'pip' version
# '--break-system-packages' allows re-installing packages which were installed by system package manager
RUN pip install pip==22.3.1 --break-system-packages

# Install packages for application
# '--no-cache-dir' - avoid using cached package information
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into application folder
COPY . /Karaburma/

# Установка PYTHONPATH
ENV PYTHONPATH=/Karaburma

# For mapping ports
# This is not for forwarding ports!
EXPOSE 8900

# Устанавливаем переменные среды на основе аргументов
#ENV HOST=0.0.0.0
#ENV PORT=8900

ENV DISPLAY=:99

# Save log files after removing container
#VOLUME ["/Karaburma/logs/"]

CMD ["bash", "-c", "Xvfb :99 -screen 0 1024x768x24 & python3 /Karaburma/karaburma/api/karaburma_api.py"]
#CMD ["python3", "/code/karaburma/api/karaburma_api.py"]
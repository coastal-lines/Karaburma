run:
    docker run -p 8900:8900 --name karaburmademo -e HOST=0.0.0.0 -e PORT=8900 -e SOURCE_MODE=screenshot -e DETECTION_MODE=default -e LOGGING=False kardemo
stop:
    docker stop karaburmademo
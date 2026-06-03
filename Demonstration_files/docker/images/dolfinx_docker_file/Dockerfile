FROM dolfinx/dolfinx:stable

RUN apt-get update && \
    apt-get install -y nano && \
    rm -rf /var/lib/apt/lists/*
 
RUN pip install --no-cache-dir matplotlib
 
CMD ["/bin/bash"]
 

FROM fenics/stable:py3

USER root

# Optional utilities
RUN apt-get update && apt-get install -y \
    nano \
    python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]
# Dockerfile

FROM algorithm-base

# Copy source code to the appropriate directory
COPY src/ /opt/algorithm/


# Set the user for the container
USER algorithm

# Set the working directory
WORKDIR /opt/algorithm

RUN pip3 install dill

ENV RUNNINGINDOCKER=True

# Set the entry point for the container
CMD ["python3", "process.py"]

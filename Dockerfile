FROM python:3.10
RUN pip install poetry
WORKDIR /lexybot
COPY poetry.lock pyproject.toml /lexybot/
RUN poetry config virtualenvs.create false && poetry install --no-interaction --no-ansi
COPY . /lexybot
CMD python run.py 
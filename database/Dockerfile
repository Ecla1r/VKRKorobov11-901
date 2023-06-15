FROM postgres:latest
ENV POSTGRES_PASSWORD=postgres
ENV POSTGRES_USER=postgres
ENV POSTGRES_DB=correlation_test
CMD ["-p", "5433"]
COPY ../init.sql /docker-entrypoint-initdb.d/init.sql
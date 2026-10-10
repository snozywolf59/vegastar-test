FROM pgvector/pgvector:pg17 AS pgvector

FROM postgis/postgis:17-3.5

COPY --from=pgvector /usr/lib/postgresql/17/lib/vector.so /usr/lib/postgresql/17/lib/vector.so
COPY --from=pgvector /usr/share/postgresql/17/extension/vector--*.sql /usr/share/postgresql/17/extension/
COPY --from=pgvector /usr/share/postgresql/17/extension/vector.control /usr/share/postgresql/17/extension/vector.control

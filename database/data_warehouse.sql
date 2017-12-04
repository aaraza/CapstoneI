CREATE TABLE author(
    author_id   SERIAL,
    name        varchar(100),
    PRIMARY KEY(author_id)
);

CREATE TABLE publication(
    site_id SERIAL,
    name    varchar(100),
    url     varchar(500), 
    PRIMARY KEY(site_id)
);
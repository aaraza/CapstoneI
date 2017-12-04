/*
 * This is the main table from which the data warehouse will be built.
 * This table will be populated by the web scraper.
 */
CREATE TABLE articles(
    site                varchar(100),
    title               varchar(250),
    author              varchar(100),
    published_on        date,
    url                 varchar(500),
    body                text,
    newspaper_keywords  varchar(300),
    newspaper_summary   text,
    id                  serial,  
    PRIMARY KEY(id)
);

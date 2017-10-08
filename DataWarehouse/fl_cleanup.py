"""
This script evaluates whether an article is likely to be in in a foreign language.
This is done by evaluating the CCTLD of an article's URL.
More information on CCTLDs: en.wikipedia.org/wiki/Country_code_top-level_domain
A list of invalid CCTLDs are in a comma seperated text file exclude.txt
"""

import authenticate
import pandas as pd
import psycopg2


def fetch_articles(cursor):
    """
    Fetch all articles from DB
    :param cursor: postgres db connection cursor
    :return: pandas DataFrame with columns URL and ID containing all articles in DB
    """
    cursor.execute("SELECT url, id FROM articles")
    raw_data = cursor.fetchall()
    data = pd.DataFrame(raw_data)
    data.columns = ['url', 'id']
    return data


def get_invalid_urls(input_file):
    """
    Read List of invalid CCTLDs from a file
    :param input_file: Text file of format "CCTLD1, CCTLD2, CCTLD3"
    :return: [List] containing invalid CCTLDs
    """
    input_file = open(file=input_file, mode="r")
    input_list = input_file.read().splitlines()[0]
    input_file.close()
    return input_list.replace(" ", "").split(",")


def validate_url(url, invalid_urls):
    """
    Check to see if article is in a foreign language by checking its CCTLD
    :param url: URL being evaluated
    :param invalid_urls: List of invalid URLs
    :return: False if the article is written in a foreign language based on CCTLD
    """
    url_ending = url.split('//')[-1].split('/')[0].split('.')[-1]
    if url_ending not in invalid_urls:
        return True
    else:
        return False


def get_foreign_article_ids(pg_cursor, invalid_urls):
    """
    Detect foreign language articles in database
    :param pg_cursor: Cursor to Postgres connection
    :param invalid_urls: List of invalid URLs
    :return: list containing foreign language article ids
    """
    data = fetch_articles(pg_cursor)

    fl_article_id = []

    for index, row in data.iterrows():
        if validate_url(row['url'], invalid_urls) is False:
            fl_article_id.append(row['id'])

    return fl_article_id


def delete_articles(pg_connection, pg_cursor, fl_articles):
    """
    Deletes all articles that have a foreign CCTLD
    :param pg_connection: Connection to Postgres database
    :param pg_cursor: Postgres database cursor
    :param fl_articles: List of foreign language article ids
    :return:
    """
    query = "DELETE FROM articles WHERE id in (%s)"
    parameter = ', '.join(list(map(lambda x: '%s', fl_articles)))
    sql = query % parameter
    pg_cursor.execute(sql, fl_articles)
    pg_connection.commit()


def main():

    pg_connection = psycopg2.connect(
        dbname=authenticate.dbname,
        user=authenticate.user,
        password=authenticate.password,
        port=authenticate.port,
        host=authenticate.host
    )

    pg_cursor = pg_connection.cursor()

    invalid_ids = get_invalid_urls(input_file="exclude.txt")
    fl_article_ids = get_foreign_article_ids(pg_cursor, invalid_ids)

    delete_articles(pg_connection, pg_cursor, fl_article_ids)


if __name__ == "__main__":
    main()

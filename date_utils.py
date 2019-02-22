from dependencies import datetime, pytz

now_utc = datetime.datetime.now(pytz.utc)
seconds_in_day = 60 * 60 * 24


def twitter_utc_string_to_datetime(string):
    return datetime.datetime.strptime(string, '%a %b %d %H:%M:%S %z %Y')


def get_age_of_date(date):
    return (now_utc - date).total_seconds() / seconds_in_day

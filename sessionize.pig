REGISTER piggybank.jar;
DEFINE ISOToUnix org.apache.pig.piggybank.evaluation.datetime.convert.ISOToUnix();
DEFINE CSVLoader org.apache.pig.piggybank.storage.CSVLoader();

/* Load data from file */
weblogs_elb = LOAD 'data/2015_07_22_mktplace_shop_web_log_sample.log.gz' USING PigStorage(' ') AS
(timestamp:chararray, elb_name:chararray, user_ip:chararray, back_end_ip:chararray, request_processing_time:float,
backend_processing_time:float, response_processing_time:float, elb_status_code:int, backend_status_code:int,
received_bytes:int, sent_bytes:int, request_string:chararray, user_agent:chararray, ssl_cipher:chararray,
ssl_protocol:chararray);

/* Group logs into sessions; for every session: state the identifying IP, time spent, number of unique URL visits,
and their associated log entries sorted by time stamp creation. */
grouped_by_ip = GROUP weblogs_elb BY user_ip;
sessions = FOREACH grouped_by_ip {
                ordered_by_ts = ORDER weblogs_elb BY timestamp;
                time_spent = ISOToUnix(MAX(weblogs_elb.timestamp)) - ISOToUnix(MIN(weblogs_elb.timestamp));
                urls = FOREACH weblogs_elb GENERATE request_string;
                distinct_urls = DISTINCT urls;
                uniq_url_visits = COUNT(distinct_urls);
                GENERATE group, time_spent, uniq_url_visits, ordered_by_ts;
}

/* Store average session time: Answers Q3 of Processing & Analytical Goals */
sessions_group = GROUP sessions ALL;
average_session_time = FOREACH sessions_group GENERATE AVG(sessions.$1);
STORE average_session_time INTO 'data/average_session_time';

/* Store sessions ordered by greatest time spent, i.e. IPs with the longest session times placed at the beginning of
file: Answers Q1, Q2, Q4 of Processing & Analytical Goals */
sessions_ordered_by_time_spent = ORDER sessions BY $1 DESC;
STORE sessions_ordered_by_time_spent INTO 'data/sessions' USING PigStorage(' ');

/* Answers Q1 for MLE Candidates: Tests for average case using Poisson Distribution to model server load. Assumptions that have been made are 
a) Occurrences of URL requests are Bernoulli random variables. b) The time period we are analyzing can be divided into many smaller sub-periods, 
ie hour/minutes/seconds/milliseconds/etc. c) The probability of two or more occurrences of a URL request within some sub-period is negligible. 
d) The probability of an occurrence is random, amortizing to a constant value within a specified time period. */
/* The formula for a Poisson Distribution is given as a follows: f(x,r,t) = exp(-rt)*(rt)*k/k! . Represents the probability distribution of 
x request arrivals in time t given an average estimated rate of arrival of r per unit time t. */
timesorted_weblogs = ORDER weblogs_elb BY timestamp; 
group_by_timesorted_weblogs = GROUP timesorted_weblogs BY ALL;
r_and_t = FOREACH group_by_timesorted_weblogs {
                GENERATE COUNT(timesorted_weblogs.$0) / ((ISOToUnix(MAX(weblogs_elb.timestamp)) - ISOToUnix(MIN(weblogs_elb.timestamp))) * .001); /* Converts ms to s: 1 ms = .001 s */
}
STORE r_and_t INTO 'data/poisson_distribution_parameters';

/* Data for use in Multilayer Perceptron Classifiers */
session_ip_and_time_spent = FOREACH sessions_group GENERATE sessions.$0,sessions.$1;
STORE session_ip_and_time_spent INTO 'data/session_ip_and_time_spent' USING PigStorage(','); /* Data utilized by SessionLengthPredictor.java to answer Q2 for MLE Candidates */

session_ip_and_unique_url_visits = FOREACH sessions_group GENERATE sessions.$0,sessions.$2;
STORE session_ip_and_time_spent INTO 'data/session_ip_and_unique_url_visits' USING PigStorage(','); /* Data utilized by SessionUniqueVisits.java to answer Q3 for MLE Candidates */


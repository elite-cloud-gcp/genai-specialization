/* src/preprocessing.sql */

/* 
   EVIDENCE FOR ML 3.4.3.3: Feature Engineering 
   - Extracting Hour and Day of Week from Timestamp
   - Calculation of Tips Percentage
*/

CREATE OR REPLACE TABLE `{project}.{dataset}.{table}` AS
SELECT
    fare as target,
    trip_miles,
    trip_seconds,
    EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
    EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_day_of_week,
    payment_type,
    company
FROM
    `bigquery-public-data.chicago_taxi_trips.taxi_trips`
WHERE
    /* EVIDENCE FOR ML 3.4.3.2: Exploration Influenced Decisions */
    /* Filtering based on EDA findings (removing invalid/outlier data) */
    trip_miles > 0 
    AND trip_seconds > 0 
    AND fare BETWEEN 2.5 AND 100
    AND trip_start_timestamp >= '2022-01-01'
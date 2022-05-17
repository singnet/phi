def tuple_time_series(time_series):
	tuple_series = []
	for row in time_series:
		row = tuple(row)
		tuple_series.append(row)
	return tuple_series
	

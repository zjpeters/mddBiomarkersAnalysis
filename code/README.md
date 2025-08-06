# Analysis of MDD data

## Background info

As discussed, I'm attaching the file we talked about for the analyses, along with a brief key explaining some of the non-self explanatory columns from the Excel sheet. Please use the second tab labelled "SimplifiedForFocusedAnalyses" as the name says, it's a simpler version of the original (first tab).


The focus of the analyses would be to identify characteristics that distinguish Healthy Controls (HC) vs patients with Major Depressive Disorder (MDD), as well as to explore potential immunological aspects that could support classification, diagnosis or treatment strategies within the MDD group. As the data is very heterogenous, an initial exploratory analysis would be a good starting point.


Please let me know if anything is unclear or if you'd prefer to go over it in person, I'd be happy to meet again if that would make things easier or more efficient on both ends.


Also, if possible, I'd really appreciate it if you could give us a rough idea of when we might hear something back. As I mentioned, we're trying to move things forward relatively quickly and aiming to wrap up the manuscript in the first half of the year.

'm' in column is for unknown. Replace with either NaN or 0 depending the condition (i.e. nan to be replaced with random value, 0 to represent "unknown")

In cases in the original spreadsheet where a value occurs like '<0.6', treat that as minimum value

There are a couple of columns that have "#VALUE!" instead of real value, so likely got lost in excel somewhere, especially common in column BB "Neutrophils per ul"

A few 0 values in BD and BE, "Classical per ul" and "Intermediate per ul", need to see if these are real or should be treated as nan

Replaced a "1?" with "1" in AK "Immune modulating drugs (current)"
## TO DO 

- [ ] write function to fill in nan values with randomized value between upper and lower limits
- [ ] write two versions of stats, one where nan values are ignored, the other using the randomized filled in data
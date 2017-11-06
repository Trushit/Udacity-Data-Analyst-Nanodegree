#This code has been adopted from the code obtained from Udacity helpform. Code was written by a forum mentor called Myles.
#This code uses sqlite3 module to import SQL functionality in python. I used it since I was working on jupyter notebook and 
#found this module an convinent way to keep everything together.

import sqlite3
import csv
from pprint import pprint

sqlite_file = 'mydb.db'

# Connect to the database
conn = sqlite3.connect(sqlite_file)
conn.text_factory = lambda x: unicode(x, 'utf-8', 'ignore')

# Get a cursor object
cur = conn.cursor()

# Create the table, specifying the column names and data types:
cur.execute('''
    CREATE TABLE nodes_tags(id INTEGER, key TEXT, value TEXT,type TEXT)     
''')

cur.execute('''
    CREATE TABLE nodes(id INTEGER, lat INTEGER, lon INTEGER, user TEXT, uid INTEGER, version INTEGER, changeset INTEGER, timestamp TEXT)
''')

cur.execute('''
    CREATE TABLE ways(id INTEGER, user TEXT, uid INTEGER, version TEXT, changeset INTEGER, timestamp TEXT)
''')

cur.execute('''
    CREATE TABLE ways_tags(id INTEGER, key TEXT, value TEXT)
''')

cur.execute('''
    CREATE TABLE ways_nodes(id INTEGER, node_id INTEGER, position TEXT)
''')

# commit the changes
conn.commit()

# Read in the csv file as a dictionary, format the
# data as a list of tuples:
with open('nodes_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'],i['value'], i['type']) for i in dr]

cur.executemany("INSERT INTO nodes_tags(id, key, value, type) VALUES(?, ?, ?, ?);", to_db)

with open('nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['lat'],i['lon'], i['user'],i['uid'],i['version'],i['changeset'],i['timestamp']) for i in dr]

cur.executemany("INSERT INTO nodes(id, lat, lon, user, uid, version, changeset, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?, ?);", to_db)    

with open('ways.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['user'],i['uid'], i['version'],i['changeset'],i['timestamp']) for i in dr]
    
cur.executemany("INSERT INTO ways(id, user, uid,version,changeset,timestamp) VALUES (?, ?, ?, ?, ?, ?);", to_db)

with open('ways_tags.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['key'],i['value']) for i in dr]
    

cur.executemany("INSERT INTO ways_tags(id, key, value) VALUES (?, ?, ?);", to_db)

with open('ways_nodes.csv','rb') as fin:
    dr = csv.DictReader(fin) # comma is default delimiter
    to_db = [(i['id'], i['node_id'],i['position']) for i in dr]
    
cur.executemany("INSERT INTO ways_nodes(id, node_id, position) VALUES (?, ?, ?);", to_db)


# commit the changes
conn.commit()


cur.execute("SELECT COUNT(*) FROM nodes;") 
pprint("Number of nodes")
pprint(cur.fetchall())

cur.execute("SELECT COUNT(*) FROM ways;")
pprint("Number of ways")
pprint(cur.fetchall())

cur.execute("SELECT COUNT(DISTINCT(e.uid)) \
FROM (SELECT uid FROM nodes UNION ALL SELECT uid FROM ways) e;") 
pprint("Number of unique users")
pprint(cur.fetchall())

cur.execute("SELECT e.user, COUNT(*) as num \
FROM (SELECT user FROM nodes UNION ALL SELECT user FROM ways) e \
GROUP BY e.user \
ORDER BY num DESC \
LIMIT 10")
pprint("Top 10 contributing users")
pprint(cur.fetchall())

cur.execute("SELECT value, COUNT(*) as num \
FROM nodes_tags \
WHERE key='amenity' \
GROUP BY value \
ORDER BY num DESC \
LIMIT 10;")
pprint("Top 10 common amenities in the area")
pprint(cur.fetchall())

cur.execute("SELECT value \
FROM nodes_tags \
WHERE key='sport';")
pprint(cur.fetchall())
conn.close()
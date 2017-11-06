import xml.etree.cElementTree as ET
from collections import defaultdict
import re
import pprint

OSMFILE = "BBC.osm"
street_type_re = re.compile(r'\b\S+\.?$', re.IGNORECASE)

#List of the expected street names
expected = ["Street", "Avenue", "Boulevard", "Drive", 
            "Court", "Place", "Square", "Lane", "Road", 
            "Trail", "Parkway", "Commons","Northeast", 
            "Run","Plaza","Terrace","Southeast","Northwest",
            "Southwest","Circle","North",
           "Crossing","Way","South","Pass"]

#List of expected postal codes
codes = ["24060" , "24061", "24062", "24063","24068","24073"]

#Mapping to correct incorrect street names to correct ones.
mapping = { "St": "Street",
            "St.": "Street",
            "Ave": "Avenue",
            "Rd.": "Road",
            "Rd": "Road",
           "Blvd": "Boulevard",
           "do rm": "do rm",
           "Dr.": "Drive"
            
            }


#Checks if street name is correct or not
def audit_street_type(street_types, street_name):
    m = street_type_re.search(street_name)
    if m:
        street_type = m.group()
        if street_type not in expected:
            street_types[street_type].add(street_name)
            
def is_street_name(elem):
    return (elem.attrib['k'] == "addr:street")

def is_postal_codes(elem):
    return (elem.attrib['k'] == "addr:postcode")

def audit(osmfile):
    osm_file = open(osmfile, "r")
    street_types = defaultdict(set)
    postal_codes = []
    for event, elem in ET.iterparse(osm_file, events=("start",)):
        if elem.tag == "node" or elem.tag == "way":
            for tag in elem.iter("tag"):
                if is_street_name(tag):
                    audit_street_type(street_types, tag.attrib['v'])
                if is_postal_codes(tag):
                    if tag.attrib['v'] not in codes:
                        postal_codes.append(tag.attrib['v'])
    
                
    osm_file.close()
    print postal_codes
    return street_types,postal_codes

#Function to update street names using mapping
def update_name(name, mapping):

    temp = name.split(" ")
    #print(temp)
    temp1 = len(temp)
    last = temp[-1]
    rem = temp[0:(len(temp)-1)]
    key = last
    
    if key in mapping: # Helps to ensure that we have a systematic error at hand
        name = ' '.join(rem) + " " + mapping[key]
    return name

#Updates postal code to appropriate format
def update_postals():
    postals = audit(OSMFILE)[1]
    
    for i in range(0,len(postals)):
        if "-" in postals[i]:
            postals[i]=postals[i].split("-")[0]
        elif " " in postals[i]:
            postals[i]=postals[i].split(" ")[1]
    #print postals 

#test function to check the functionality of the above code
def test():
    st_types = audit(OSMFILE)[0]
    for st_type, ways in st_types.iteritems():
        print st_type,ways
        for name in ways:
            better_name = update_name(name, mapping)
            print name, "=>", better_name
            
test()
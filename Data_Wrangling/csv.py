#Following code has been adpoted from the online lesson on udacity.com. Code has been 
#modified to correct postal code along with street names. Further, code generates the csv files
#that shall be pushed to SQL database for exploration. 


import csv
import codecs
import pprint
import re
import xml.etree.cElementTree as ET

import cerberus

import schema

OSM_PATH = "sample.osm"

NODES_PATH = "nodes.csv"
NODE_TAGS_PATH = "nodes_tags.csv"
WAYS_PATH = "ways.csv"
WAY_NODES_PATH = "ways_nodes.csv"
WAY_TAGS_PATH = "ways_tags.csv"

LOWER_COLON = re.compile(r'^([a-z]|_)+:([a-z]|_)+')
PROBLEMCHARS = re.compile(r'[=\+/&<>;\'"\?%#$@\,\. \t\r\n]')

SCHEMA = schema.schema

# Make sure the fields order in the csvs matches the column order in the sql table schema
NODE_FIELDS = ['id', 'lat', 'lon', 'user', 'uid', 'version', 'changeset', 'timestamp']
NODE_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_FIELDS = ['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
WAY_TAGS_FIELDS = ['id', 'key', 'value', 'type']
WAY_NODES_FIELDS = ['id', 'node_id', 'position']


def shape_element(element, node_attr_fields=NODE_FIELDS, way_attr_fields=WAY_FIELDS,
                  problem_chars=PROBLEMCHARS, default_tag_type='regular'):
    """Clean and shape node or way XML element to Python dict"""

    node_attribs = {}
    way_attribs = {}
    way_nodes = []
    tags = []  # Handle secondary tags the same way for both node and way elements

    # Modified code starts here
    
    if element.tag == "node":
        temp_attr={}
        attr=["id","user","uid","version","lat","lon","timestamp","changeset"]
        for attr in attr:
            temp_attr[attr]=element.attrib[attr]
        
        node_attribs = temp_attr
        
        for tag in element.iter():
            temp_tag={}
            temp_tag["id"] = element.attrib["id"]
            if tag.tag=="tag":
           
             if PROBLEMCHARS.search(tag.attrib["k"]) is None:
                    if ":" in tag.attrib["k"]:
                        x=tag.attrib["k"].split(":")
                        temp_tag["key"] = ":".join(x[1:len(x)])
                        temp_tag["type"] = x[0]
                        if tag.attrib["k"] == 'addr:street':
                            temp_tag["value"] = update_name(tag.attrib["v"], mapping)
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]
                    else:
                        temp_tag["key"] = tag.attrib["k"]
                        temp_tag["type"] = "regular"
                        if tag.attrib["k"] == 'addr:street':
                            temp_tag["value"] = update_name(tag.attrib["v"], mapping)
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]
                            
                        if tag.attrib["k"] == 'addr:postcode':
                            temp_tag["value"] = update_postals()
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]   
                    
                    tags.append(temp_tag)
    
    elif element.tag == "way":
        temp_attr1={}
        temp_attr1_iter={}
        counter=0
        attr1=['id', 'user', 'uid', 'version', 'changeset', 'timestamp']
        for attr in attr1:
            temp_attr1[attr]=element.attrib[attr]
        way_attribs = temp_attr1
        for tag in element.iter("nd"):
            
            temp_attr1_iter={}
            
            temp_attr1_iter["id"]=element.attrib["id"]
            temp_attr1_iter["node_id"] = tag.attrib["ref"]
            temp_attr1_iter["position"] = counter
            counter=counter + 1
            way_nodes.append(temp_attr1_iter)
            print(way_nodes)
        
        for tag in element.iter():
            temp_tag={}
            temp_tag["id"] = element.attrib["id"]
            if tag.tag=="tag":
           
             if PROBLEMCHARS.search(tag.attrib["k"]) is None:
                    if ":" in tag.attrib["k"]:
                        x=tag.attrib["k"].split(":")
                        temp_tag["key"] = ":".join(x[1:len(x)])
                        temp_tag["type"] = x[0]
                        if tag.attrib["k"] == 'addr:street':
                            temp_tag["value"] = update_name(tag.attrib["v"], mapping)
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]
                        
                    
                        if tag.attrib["k"] == 'addr:postcode':
                            temp_tag["value"] = update_postals()
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]   
                    
                    else:
                        temp_tag["key"] = tag.attrib["k"]
                        temp_tag["type"] = "regular" 
                        if tag.attrib["k"] == 'addr:street':
                            temp_tag["value"] = update_name(tag.attrib["v"], mapping)
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]
                        
                        if tag.attrib["k"] == 'addr:postcode':
                            temp_tag["value"] = update_postals()
                    # otherwise:
                        else:
                            temp_tag["value"] = tag.attrib["v"]   
                    
                    tags.append(temp_tag)
    
    if element.tag == 'node':
        return {'node': node_attribs, 'node_tags': tags}
    elif element.tag == 'way':
        return {'way': way_attribs, 'way_nodes': way_nodes, 'way_tags': tags}


# ================================================== #
#               Helper Functions                     #
# ================================================== #
def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag"""

    context = ET.iterparse(osm_file, events=('start', 'end'))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


def validate_element(element, validator, schema=SCHEMA):
    """Raise ValidationError if element does not match schema"""
    
    if validator.validate(element, schema) is not True:
        field, errors = next(validator.errors.iteritems())
        message_string = "\nElement of type '{0}' has the following errors:\n{1}"
        error_string = pprint.pformat(errors)
        
        raise Exception(message_string.format(field, error_string))


class UnicodeDictWriter(csv.DictWriter, object):
    """Extend csv.DictWriter to handle Unicode input"""

    def writerow(self, row):
        super(UnicodeDictWriter, self).writerow({
            k: (v.encode('utf-8') if isinstance(v, unicode) else v) for k, v in row.iteritems()
        })

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


# ================================================== #
#               Main Function                        #
# ================================================== #
def process_map(file_in, validate):
    """Iteratively process each XML element and write to csv(s)"""

    with codecs.open(NODES_PATH, 'w') as nodes_file, \
         codecs.open(NODE_TAGS_PATH, 'w') as nodes_tags_file, \
         codecs.open(WAYS_PATH, 'w') as ways_file, \
         codecs.open(WAY_NODES_PATH, 'w') as way_nodes_file, \
         codecs.open(WAY_TAGS_PATH, 'w') as way_tags_file:

        nodes_writer = UnicodeDictWriter(nodes_file, NODE_FIELDS)
        node_tags_writer = UnicodeDictWriter(nodes_tags_file, NODE_TAGS_FIELDS)
        ways_writer = UnicodeDictWriter(ways_file, WAY_FIELDS)
        way_nodes_writer = UnicodeDictWriter(way_nodes_file, WAY_NODES_FIELDS)
        way_tags_writer = UnicodeDictWriter(way_tags_file, WAY_TAGS_FIELDS)

        nodes_writer.writeheader()
        node_tags_writer.writeheader()
        ways_writer.writeheader()
        way_nodes_writer.writeheader()
        way_tags_writer.writeheader()

        validator = cerberus.Validator()

        for element in get_element(file_in, tags=('node', 'way')):
            el = shape_element(element)
            if el:
                if validate is True:
                    validate_element(el, validator)

                if element.tag == 'node':
                    nodes_writer.writerow(el['node'])
                    node_tags_writer.writerows(el['node_tags'])
                elif element.tag == 'way':
                    ways_writer.writerow(el['way'])
                    way_nodes_writer.writerows(el['way_nodes'])
                    way_tags_writer.writerows(el['way_tags'])


if __name__ == '__main__':
    # Note: Validation is ~ 10X slower. For the project consider using a small
    # sample of the map when validating.
    process_map(OSM_PATH, validate=True)

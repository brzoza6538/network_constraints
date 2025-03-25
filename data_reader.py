import re

def parse_nodes(section):
    nodes = {}
    pattern = r"(\w+)\s*\(\s*([\d\.]+)\s+([\d\.]+)\s*\)"
    #np. Gdansk ( 18.60 54.20 )

    matches = re.findall(pattern, section)
    for match in matches:
        node_id = match[0]
        longitude = float(match[1])
        latitude = float(match[2])
        nodes[node_id] = {"longitude": longitude, "latitude": latitude}
    return nodes

def parse_links(section):
    links = {}
    pattern = r"(\w+)\s*\(\s*(\w+)\s+(\w+)\s*\)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*([\d\.]+)\s*\((.*?)\)"
    #np. Link_0_10 ( Gdansk Warsaw ) 0.00 0.00 0.00 156.00 ( 155.00 156.00 622.00 468.00 )
   
    matches = re.findall(pattern, section)
    for match in matches:
        link_id = match[0]
        source = match[1]
        target = match[2]
        pre_installed_capacity = int(round(float(match[3])))
        pre_installed_capacity_cost = int(round(float(match[4])))
        routing_cost = int(round(float(match[5])))
        setup_cost = int(round(float(match[6])))
        modules = match[7].strip().split()
        modules = [(int(round(float(modules[i]))), int(round(float(modules[i + 1])))) for i in range(0, len(modules), 2)]
        links[link_id] = {
            "source": source,
            "target": target,
            "pre_installed_capacity": pre_installed_capacity,
            "pre_installed_capacity_cost": pre_installed_capacity_cost,
            "routing_cost": routing_cost,
            "setup_cost": setup_cost,
            "modules": modules
        }
    return links

def parse_demands(section):
    demands = {}
    pattern = r"(\w+)\s*\(\s*(\w+)\s+(\w+)\s*\)\s*(\d+)\s*([\d\.]+)\s*(\w+)"
    #np Demand_2_6 ( Kolobrzeg Lodz ) 1 128.00 UNLIMITED

    matches = re.findall(pattern, section)
    for match in matches:
        demand_id = match[0]
        source = match[1]
        target = match[2]
        routing_unit = int(match[3])
        demand_value = int(round(float(match[4])))
        max_path_length = match[5]
        demands[demand_id] = {
            "source": source,
            "target": target,
            "routing_unit": routing_unit,
            "demand_value": demand_value,
            "max_path_length": max_path_length
        }
    return demands

def parse_admissible_paths(text):
    paths = {}
    current_demand = None
    
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Demand"):
            current_demand = re.match(r"Demand_\d+_\d+", line).group(0)
            paths[current_demand] = {}
        elif line.startswith("P_") and current_demand:
            path_match = re.match(r"P_(\d+)\s+\(([^)]+)\)", line)
            if path_match:
                path_id = f"P_{path_match.group(1)}"
                links = path_match.group(2).split()
                paths[current_demand][path_id] = links
    return paths

def parse_sndlib_file(file_content):
    sections = re.split(r"(?=#\s)", file_content)
    nodes, links, demands, admissible_paths = {}, {}, {}, {}

    for section in sections:
        if "NODES" in section:
            nodes = parse_nodes(section)
        elif "LINKS" in section:
            links = parse_links(section)
        elif "DEMANDS" in section:
            demands = parse_demands(section)
        elif "ADMISSIBLE_PATHS" in section:
            admissible_paths = parse_admissible_paths(section)
    return nodes, links, demands, admissible_paths


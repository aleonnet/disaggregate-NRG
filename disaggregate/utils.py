def show_top_devices(redd_data, n_top=5):
    ''' produce list of n_top power-using appliances for each building
    '''
    tops = []
    for i in range(1,7):
        tmp = redd_data.buildings[i]
        tops.append(tmp.elec.submeters().select_top_k(k=n_top))
    print(tops)


def subset_present_types(building, all_types):
    ''' search a specific building for appliance types present in the
    master list. returns a list of the found types.

    Input:
        building: a complete building instance

        all_types: a list of appliances to search for
    '''
    valid_list = []
    for item in all_types:
        try:
            building.elec.select_using_appliances(type=item)
            valid_list.append(item)
        except KeyError:
            continue

    return valid_list


def number_list_duplicates(in_list):
    newlist = []
    for i, v in enumerate(in_list):
        totalcount = in_list.count(v)
        count = in_list[:i].count(v)
        newlist.append(v + '.' + str(count + 1) if totalcount > 1 else v)
    return newlist

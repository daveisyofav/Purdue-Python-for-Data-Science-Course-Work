import re

def problem1(searchstring):
    """
    Match phone numbers.

    :param searchstring: string
    :return: True or False
    """
    #Check for country code numbers that include spaces and dashes and optionally parentheses
    countryCode = re.compile(r'\+(1 |52 )(\(?)(\d\d\d)(\)?)( |\-)(\d\d\d)(-)(\d\d\d\d)')
    countryCodeFull = countryCode.fullmatch(searchstring)

    if countryCodeFull:
        # In this circumstance, double check that if one parenthesis is there, the other is too
        if bool(countryCodeFull.group(2)) == bool(countryCodeFull.group(4)):
            #Also check that if there IS parentheses, there is NOT a dash and vice versa
            if (bool(countryCodeFull.group(2)) and (countryCodeFull.group(5) == ' ')) or (not(bool(countryCodeFull.group(2))) and (countryCodeFull.group(5) == '-')):
                return True

    # a case not covered yet is if there is a country code followed by only digits:
    simpleCC = re.compile(r'\+(1 |52 )(\d\d\d\d\d\d\d\d\d\d)')
    if simpleCC.fullmatch(searchstring):
        return True

    # Now cover when there isn't a country code, in which case it has to be 3 digits a dash then 4 digits
    localCall = re.compile(r'\d\d\d-\d\d\d\d')
    if localCall.fullmatch(searchstring):
        return True

    return False

        
def problem2(searchstring):
    """
    Extract street name from address.

    :param searchstring: string
    :return: string
    """
    validAddy_R = re.compile(r'(\d+)(( [A-Z]([a-z])*)+)( Rd\.| Dr\.| Ave\.| St\.)')
    validAddy_M = validAddy_R.search(searchstring)

    #Remove the road type off the end!
    FinalAddy = validAddy_M.group(1) + validAddy_M.group(2)
    #print(FinalAddy)
    return FinalAddy
    
def problem3(searchstring):
    """
    Garble Street name.

    :param searchstring: string
    :return: string
    """
    validAddy_R = re.compile(r'(\d+)(( [A-Z]([a-z])*)+)( Rd\.| Dr\.| Ave\.| St\.)')
    doorNum = validAddy_R.search(searchstring).group(1)
    Streetname = validAddy_R.search(searchstring).group(2)
    roadType = validAddy_R.search(searchstring).group(5)
    #print('roadtype = ' + roadType)

    bckwrdsStreetname = Streetname[::-1]
    #print(bckwrdsStreetname)

    bckwrdsAddy = doorNum + ' ' + bckwrdsStreetname + roadType[1:]  #a couple of space issues are fixed here

    bckwrdsFullString = validAddy_R.sub(bckwrdsAddy, searchstring)

    print(bckwrdsFullString)
    return bckwrdsFullString

if __name__ == '__main__' :
    print("\nProblem 1:")
    print("Answer correct?",problem1('+1 765-494-4600')==True)
    print("Answer correct?",problem1('+52 765-494-4600 ')==False)
    print("Answer correct?",problem1('+1 (765) 494 4600')==False)
    print("Answer correct?",problem1('+52 (765) 494-4600')==True)
    print("Answer correct?",problem1('+52 7654944600')==True)
    print("Answer correct?",problem1('494-4600')==True)

    print("\nProblem 2:")
    print("Answer correct?",problem2('The EE building is at 465 Northwestern Ave.')=="465 Northwestern")
    print("Answer correct?",problem2('Meet me at 201 South First St. at noon')=="201 South First")
    print("Answer correct?",problem2('Type "404 Not Found St" on your phone at 201 South First St. at noon')=="201 South First")

    print("\nProblem 3:")
    print("Answer correct?",problem3('The EE building is at 465 Northwestern Ave.')=="The EE building is at 465 nretsewhtroN Ave.")
    print("Answer correct?",problem3('Meet me at 201 South First St. at noon')=="Meet me at 201 tsriF htuoS St. at noon")

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 18:28:04 2015
@author: gfleishman
"""

import pyrpl.interface.parse as parse
import pyrpl.optimization.optimizer as optimizer


def main():

    # get registration specifications
    input_dictionary = parse.parse_input_commands()
    output_dictionary = parse.parse_output_commands()

    # report back registration specifications
    print '\nREGISTRATION SPECIFICATIONS:'
    for key in input_dictionary.keys():
        print key + ":\t\t\t" + str(input_dictionary[key])
    print '\nWRITE SPECIFICATIONS:'
    for key in output_dictionary.keys():
        print key + ":\t\t\t" + str(output_dictionary[key])
    print '\n\n\n'

    # begin optimization
    result = optimizer.optimize(input_dictionary)

    # write the output
    parse.write_output(input_dictionary, output_dictionary, result)

main()

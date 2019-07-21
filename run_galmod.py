""" Main script to run the galactic model"""
import default_samplers

def main():
    """ Simple main function example"""
    output = default_samplers.NestedSampling(\
        '../Binaryfiles/Renorm6ext2/renormalised_0-samples-production.npy')
    output.drun()


if __name__ == "__main__":
    main()

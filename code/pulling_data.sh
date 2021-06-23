aws ec2 describe-spot-price-history --output table\
                                    --product-description "Linux/UNIX (Amazon VPC)"\
                                    --region us-west-1 > ../data/us_west_1.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "Linux/UNIX (Amazon VPC)"\
                                    --region us-west-1 > ../data/us_west_2.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "Linux/UNIX (Amazon VPC)"\
                                    --region us-west-1 > ../data/us_east_1.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "Linux/UNIX (Amazon VPC)"\
                                    --region us-west-1 > ../data/us_east_2.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "SUSE Linux"\
                                    --region us-west-1 > ../data/us_west_1_linux.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "SUSE Linux"\
                                    --region us-west-1 > ../data/us_west_2_linux.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "SUSE Linux"\
                                    --region us-west-1 > ../data/us_east_1_linux.txt

aws ec2 describe-spot-price-history --output table\
                                    --product-description "SUSE Linux"\
                                    --region us-west-1 > ../data/us_east_2_linux.txt

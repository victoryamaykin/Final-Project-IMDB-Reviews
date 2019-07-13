# from sqlalchemy import create_engine

# from sqlalchemy.ext.declarative import declarative_base

# from sqlalchemy import Column, Integer, String, Float 

# # Sets an object to utilize the default declarative base in SQL Alchemy
# Base = declarative_base()

# class Review(Base):
#     __tablename__ = 'reviews'

#     id = Column(Integer, primary_key=True)
#     review = Column(String(255))
    

# engine = create_engine("sqlite:///db/reviews.sqlite")
# conn = engine.connect()

# Base.metadata.create_all(engine)
# # Base.metadata.drop_all(engine) 

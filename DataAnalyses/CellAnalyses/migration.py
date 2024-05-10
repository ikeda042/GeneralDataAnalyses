from sqlalchemy import Integer, String, BLOB, FLOAT, create_engine, Column
from sqlalchemy.orm import declarative_base
from numpy.linalg import inv
from sqlalchemy import create_engine, Column, Integer, String, BLOB, FLOAT
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


Base2 = declarative_base()


class Cell(Base2):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB)
    contour = Column(BLOB)


Base = declarative_base()


class Cell2(Base):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB)
    img_fluo2 = Column(BLOB, nullable=True)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)
    max_brightness = Column(FLOAT)
    min_brightness = Column(FLOAT)
    mean_brightness_raw = Column(FLOAT)
    mean_brightness_normalized = Column(FLOAT)
    median_brightness_raw = Column(FLOAT)
    median_brightness_normalized = Column(FLOAT)
    ph_max_brightness = Column(FLOAT)
    ph_min_brightness = Column(FLOAT)
    ph_mean_brightness_raw = Column(FLOAT)
    ph_mean_brightness_normalized = Column(FLOAT)
    ph_median_brightness_raw = Column(FLOAT)
    ph_median_brightness_normalized = Column(FLOAT)


Base3 = declarative_base()

class Cell3(Base3):
    __tablename__ = "cells"
    id = Column(Integer, primary_key=True)
    cell_id = Column(String)
    label_experiment = Column(String)
    manual_label = Column(Integer)
    perimeter = Column(FLOAT)
    area = Column(FLOAT)
    img_ph = Column(BLOB)
    img_fluo1 = Column(BLOB)
    # img_fluo2 = Column(BLOB)
    contour = Column(BLOB)
    center_x = Column(FLOAT)
    center_y = Column(FLOAT)

# engine = create_engine("sqlite:///sk25_LB_3ml_1.db")
# Base3.metadata.create_all(engine)
# Session = sessionmaker(bind=engine)
# session = Session()
# cells = [cell for cell in session.query(Cell3).all() if cell.manual_label == "N/A"]


# engine2 = create_engine("sqlite:///sk25_LB_3ml_negative.db")
# Base2.metadata.create_all(engine2)
# Session2 = sessionmaker(bind=engine2)
# session2 = Session2()


# for cell in cells:
#     cell2 = Cell(
#         cell_id=cell.cell_id,
#         img_ph=cell.img_ph,
#         img_fluo1=cell.img_fluo1,
#         contour=cell.contour,
#     )
#     session2.add(cell2)
# session2.commit()


def extract_NA(db_path:str) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    Base3.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    cells = [cell for cell in session.query(Cell3).all() if cell.manual_label == "N/A"]

    new_db_path = f"{db_path.split('.')[0]}_NA.db"
    if os.path.exists(new_db_path):
        print(f"Database {new_db_path} already exists. Skipping...")
        return
    
    engine2 = create_engine(f"sqlite:///{db_path.split(".")[0]}_NA.db")
    Base2.metadata.create_all(engine2)
    Session2 = sessionmaker(bind=engine2)
    session2 = Session2()
    for cell in cells:
        cell2 = Cell(
            cell_id=cell.cell_id,
            img_ph=cell.img_ph,
            img_fluo1=cell.img_fluo1,
            contour=cell.contour,
        )
        session2.add(cell2)
    session2.commit()


def extract_1(db_path:str) -> None:
    engine = create_engine(f"sqlite:///{db_path}")
    Base3.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    cells = [cell for cell in session.query(Cell3).all() if cell.manual_label == 1]

    new_db_path = f"{db_path.split('.')[0]}_1.db"
    if os.path.exists(new_db_path):
        print(f"Database {new_db_path} already exists. Skipping...")
        return

    engine2 = create_engine(f"sqlite:///{new_db_path}")
    Base2.metadata.create_all(engine2)
    Session2 = sessionmaker(bind=engine2)
    session2 = Session2()
    for cell in cells:
        cell2 = Cell(
            cell_id=cell.cell_id,
            img_ph=cell.img_ph,
            img_fluo1=cell.img_fluo1,
            contour=cell.contour,
        )
        session2.add(cell2)
    session2.commit()

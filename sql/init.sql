-- SCHEMA: image_processing

CREATE SCHEMA IF NOT EXISTS image_processing
    AUTHORIZATION gpig;


-- Table: image_processing.raw_entry

CREATE TABLE IF NOT EXISTS image_processing.raw_entry
(
    image_uri varchar COLLATE pg_catalog."default" NOT NULL,
    latitude real NOT NULL,
    longitude real NOT NULL,
    date timestamp NOT NULL,
    seen BOOLEAN DEFAULT 'f',
    CONSTRAINT raw_entry_pkey PRIMARY KEY (image_uri)
);

ALTER TABLE image_processing.raw_entry
    OWNER to gpig;

-- Table: image_processing.processed_entry

CREATE TABLE IF NOT EXISTS image_processing.processed_entry
(
    id serial NOT NULL,
    image_uri varchar COLLATE pg_catalog."default" NOT NULL,
    plant_id varchar COLLATE pg_catalog."default",
    CONSTRAINT processed_entry_pkey PRIMARY KEY (id),
    CONSTRAINT processed_entry_raw_entry_image_uri_fk FOREIGN KEY (image_uri)
        REFERENCES image_processing.raw_entry (image_uri) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
);

ALTER TABLE image_processing.processed_entry
    OWNER to gpig;

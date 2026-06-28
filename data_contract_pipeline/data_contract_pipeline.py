import yaml
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_date, to_timestamp
from pyspark.sql.types import (
    StringType,
    IntegerType,
    LongType,
    DoubleType,
    FloatType,
    DecimalType,
    BooleanType,
    DateType,
    TimestampType,
)


class DataContractError(Exception):
    """Raised when one or more data contract checks fail."""


# ============================================================
# YAML loading
# ============================================================

def load_yaml_contract(contract_path: str) -> dict:
    with open(contract_path, "r") as file:
        return yaml.safe_load(file)


# ============================================================
# Contract helpers
# ============================================================

def get_expected_columns(contract: dict) -> list[str]:
    columns = contract.get("columns", [])
    ordered_columns = sorted(
        columns,
        key=lambda column: column.get("ordinal_position", 999999)
    )
    return [column["name"] for column in ordered_columns]


def get_column_type_map(contract: dict) -> dict:
    return {
        column["name"]: column["type"]
        for column in contract.get("columns", [])
        if "name" in column and "type" in column
    }


def get_column_format_map(contract: dict) -> dict:
    return {
        column["name"]: column["format"]
        for column in contract.get("columns", [])
        if "name" in column and "format" in column
    }


def get_primary_key_columns(contract: dict) -> list[str]:
    if "primary_key" in contract:
        return contract["primary_key"]

    table_config = contract.get("table", {})
    if "primary_key" in table_config:
        return table_config["primary_key"]

    return [
        column["name"]
        for column in contract.get("columns", [])
        if column.get("key_part") is True
    ]


def get_spark_type(type_name: str):
    clean_type = type_name.lower().strip()

    type_mapping = {
        "string": StringType(),
        "integer": IntegerType(),
        "int": IntegerType(),
        "long": LongType(),
        "bigint": LongType(),
        "double": DoubleType(),
        "float": FloatType(),
        "decimal": DecimalType(18, 2),
        "boolean": BooleanType(),
        "date": DateType(),
        "timestamp": TimestampType(),
    }

    if clean_type not in type_mapping:
        raise DataContractError(f"Unsupported data type in contract: {type_name}")

    return type_mapping[clean_type]


def build_table_name(table_config: dict) -> str:
    return (
        f"{table_config['catalog']}."
        f"{table_config['schema']}."
        f"{table_config['table']}"
    )


def build_schema_name(table_config: dict) -> str:
    return f"{table_config['catalog']}.{table_config['schema']}"


# ============================================================
# Validation checks
# ============================================================

def validate_schema(df: DataFrame, contract: dict) -> list[str]:
    errors = []

    actual_columns = df.columns
    expected_columns = get_expected_columns(contract)

    schema_rules = contract.get("schema_enforcement", {})
    expected_field_count = schema_rules.get("expected_field_count", len(expected_columns))
    allow_extra_columns = schema_rules.get("allow_extra_columns", False)
    allow_missing_columns = schema_rules.get("allow_missing_columns", False)
    enforce_column_order = schema_rules.get("enforce_column_order", False)
    case_sensitive_columns = schema_rules.get("case_sensitive_columns", True)

    if not case_sensitive_columns:
        actual_compare = [column.lower() for column in actual_columns]
        expected_compare = [column.lower() for column in expected_columns]
    else:
        actual_compare = actual_columns
        expected_compare = expected_columns

    if len(actual_columns) != expected_field_count:
        errors.append(
            f"Field count mismatch. Expected {expected_field_count}, "
            f"but found {len(actual_columns)}."
        )

    missing_columns = set(expected_compare) - set(actual_compare)
    if missing_columns and not allow_missing_columns:
        errors.append(f"Missing columns: {sorted(missing_columns)}.")

    extra_columns = set(actual_compare) - set(expected_compare)
    if extra_columns and not allow_extra_columns:
        errors.append(f"Unexpected columns: {sorted(extra_columns)}.")

    if enforce_column_order and actual_compare != expected_compare:
        errors.append("Column order mismatch. Source column order does not match contract.")

    return errors


def validate_date_and_timestamp_formats(df: DataFrame, contract: dict) -> list[str]:
    errors = []
    format_map = get_column_format_map(contract)
    type_map = get_column_type_map(contract)

    for column_name, expected_format in format_map.items():
        if column_name not in df.columns:
            errors.append(f"Formatted column '{column_name}' is missing.")
            continue

        column_type = type_map.get(column_name, "string").lower().strip()

        if column_type == "date":
            invalid_count = (
                df.filter(
                    col(column_name).isNotNull()
                    & to_date(col(column_name).cast("string"), expected_format).isNull()
                )
                .count()
            )
        elif column_type == "timestamp":
            invalid_count = (
                df.filter(
                    col(column_name).isNotNull()
                    & to_timestamp(col(column_name).cast("string"), expected_format).isNull()
                )
                .count()
            )
        else:
            continue

        if invalid_count > 0:
            errors.append(
                f"Format check failed for '{column_name}'. "
                f"Expected format: {expected_format}. "
                f"Invalid records: {invalid_count}."
            )

    return errors


def validate_nullable_rules(df: DataFrame, contract: dict) -> list[str]:
    errors = []

    for column_def in contract.get("columns", []):
        column_name = column_def["name"]
        nullable = column_def.get("nullable", True)

        if nullable is False and column_name in df.columns:
            null_count = df.filter(col(column_name).isNull()).count()

            if null_count > 0:
                errors.append(
                    f"Null check failed for '{column_name}'. "
                    f"Null records found: {null_count}."
                )

    return errors


def validate_primary_key_rules(df: DataFrame, contract: dict) -> list[str]:
    errors = []
    primary_keys = get_primary_key_columns(contract)

    if not primary_keys:
        return errors

    missing_keys = [key for key in primary_keys if key not in df.columns]
    if missing_keys:
        errors.append(f"Primary key columns missing: {missing_keys}.")
        return errors

    null_key_condition = " OR ".join([f"`{key}` IS NULL" for key in primary_keys])
    null_key_count = df.filter(null_key_condition).count()

    if null_key_count > 0:
        errors.append(
            f"Primary key null check failed. "
            f"Rows with null primary key values: {null_key_count}."
        )

    duplicate_key_count = (
        df.groupBy(*primary_keys)
        .count()
        .filter(col("count") > 1)
        .count()
    )

    if duplicate_key_count > 0:
        errors.append(
            f"Primary key uniqueness check failed. "
            f"Duplicate key combinations found: {duplicate_key_count}."
        )

    return errors


def validate_data_contract(df: DataFrame, contract: dict) -> None:
    errors = []

    errors.extend(validate_schema(df, contract))
    errors.extend(validate_date_and_timestamp_formats(df, contract))
    errors.extend(validate_nullable_rules(df, contract))
    errors.extend(validate_primary_key_rules(df, contract))

    if errors:
        raise DataContractError(
            "Data contract validation failed:\n- " + "\n- ".join(errors)
        )


# ============================================================
# Transformation and write
# ============================================================

def select_contract_columns(df: DataFrame, contract: dict) -> DataFrame:
    expected_columns = get_expected_columns(contract)
    return df.select(*[col(column_name) for column_name in expected_columns])


def apply_column_types(df: DataFrame, contract: dict) -> DataFrame:
    column_type_map = get_column_type_map(contract)
    format_map = get_column_format_map(contract)

    for column_name, type_name in column_type_map.items():
        if column_name not in df.columns:
            raise DataContractError(f"Cannot cast missing column: {column_name}")

        spark_type = get_spark_type(type_name)

        if isinstance(spark_type, DateType):
            date_format = format_map.get(column_name, "yyyyMMdd")
            df = df.withColumn(
                column_name,
                to_date(col(column_name).cast("string"), date_format)
            )
        elif isinstance(spark_type, TimestampType):
            timestamp_format = format_map.get(column_name, "yyyy-MM-dd HH:mm:ss")
            df = df.withColumn(
                column_name,
                to_timestamp(col(column_name).cast("string"), timestamp_format)
            )
        else:
            df = df.withColumn(column_name, col(column_name).cast(spark_type))

    return df


def run_data_contract_pipeline(
    spark: SparkSession,
    contract_path: str,
    write_mode: str = "overwrite"
) -> None:
    print(f"Loading data contract: {contract_path}")
    contract = load_yaml_contract(contract_path)

    source_table = build_table_name(contract["source"])
    target_table = build_table_name(contract["target"])
    target_schema = build_schema_name(contract["target"])

    print(f"Reading source table: {source_table}")
    source_df = spark.table(source_table)

    print("Running data contract validation")
    validate_data_contract(source_df, contract)
    print("Data contract validation passed")

    print("Selecting contract columns")
    contracted_df = select_contract_columns(source_df, contract)

    print("Applying target data types")
    typed_df = apply_column_types(contracted_df, contract)

    print(f"Creating target schema if needed: {target_schema}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {target_schema}")

    print(f"Writing target Delta table: {target_table}")
    (
        typed_df.write
        .format("delta")
        .mode(write_mode)
        .option("overwriteSchema", "true")
        .saveAsTable(target_table)
    )

    print(f"Pipeline completed successfully: {target_table}")

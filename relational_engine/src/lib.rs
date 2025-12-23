use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tensor_store::{ScalarValue, TensorData, TensorStore, TensorStoreError, TensorValue};

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    Int,
    Float,
    String,
    Bool,
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub column_type: ColumnType,
    pub nullable: bool,
}

impl Column {
    pub fn new(name: impl Into<String>, column_type: ColumnType) -> Self {
        Self {
            name: name.into(),
            column_type,
            nullable: false,
        }
    }

    pub fn nullable(mut self) -> Self {
        self.nullable = true;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn new(columns: Vec<Column>) -> Self {
        Self { columns }
    }

    pub fn get_column(&self, name: &str) -> Option<&Column> {
        self.columns.iter().find(|c| c.name == name)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl Value {
    fn to_scalar(&self) -> ScalarValue {
        match self {
            Value::Null => ScalarValue::Null,
            Value::Int(v) => ScalarValue::Int(*v),
            Value::Float(v) => ScalarValue::Float(*v),
            Value::String(v) => ScalarValue::String(v.clone()),
            Value::Bool(v) => ScalarValue::Bool(*v),
        }
    }

    fn from_scalar(scalar: &ScalarValue) -> Self {
        match scalar {
            ScalarValue::Null => Value::Null,
            ScalarValue::Int(v) => Value::Int(*v),
            ScalarValue::Float(v) => Value::Float(*v),
            ScalarValue::String(v) => Value::String(v.clone()),
            ScalarValue::Bool(v) => Value::Bool(*v),
            ScalarValue::Bytes(_) => Value::Null,
        }
    }

    fn matches_type(&self, column_type: &ColumnType) -> bool {
        matches!(
            (self, column_type),
            (Value::Null, _)
                | (Value::Int(_), ColumnType::Int)
                | (Value::Float(_), ColumnType::Float)
                | (Value::String(_), ColumnType::String)
                | (Value::Bool(_), ColumnType::Bool)
        )
    }
}

#[derive(Debug, Clone)]
pub struct Row {
    pub id: u64,
    pub values: HashMap<String, Value>,
}

impl Row {
    pub fn get(&self, column: &str) -> Option<&Value> {
        if column == "_id" {
            return None;
        }
        self.values.get(column)
    }

    pub fn get_with_id(&self, column: &str) -> Option<Value> {
        if column == "_id" {
            return Some(Value::Int(self.id as i64));
        }
        self.values.get(column).cloned()
    }
}

#[derive(Debug, Clone)]
pub enum Condition {
    Eq(String, Value),
    Ne(String, Value),
    Lt(String, Value),
    Le(String, Value),
    Gt(String, Value),
    Ge(String, Value),
    And(Box<Condition>, Box<Condition>),
    Or(Box<Condition>, Box<Condition>),
    True,
}

impl Condition {
    pub fn and(self, other: Condition) -> Condition {
        Condition::And(Box::new(self), Box::new(other))
    }

    pub fn or(self, other: Condition) -> Condition {
        Condition::Or(Box::new(self), Box::new(other))
    }

    fn evaluate(&self, row: &Row) -> bool {
        match self {
            Condition::True => true,
            Condition::Eq(col, val) => row.get_with_id(col).as_ref() == Some(val),
            Condition::Ne(col, val) => row.get_with_id(col).as_ref() != Some(val),
            Condition::Lt(col, val) => self.compare_lt(row, col, val),
            Condition::Le(col, val) => self.compare_le(row, col, val),
            Condition::Gt(col, val) => self.compare_gt(row, col, val),
            Condition::Ge(col, val) => self.compare_ge(row, col, val),
            Condition::And(a, b) => a.evaluate(row) && b.evaluate(row),
            Condition::Or(a, b) => a.evaluate(row) || b.evaluate(row),
        }
    }

    fn compare_lt(&self, row: &Row, col: &str, val: &Value) -> bool {
        match (row.get_with_id(col), val) {
            (Some(Value::Int(a)), Value::Int(b)) => a < *b,
            (Some(Value::Float(a)), Value::Float(b)) => a < *b,
            (Some(Value::String(a)), Value::String(b)) => a < *b,
            _ => false,
        }
    }

    fn compare_le(&self, row: &Row, col: &str, val: &Value) -> bool {
        match (row.get_with_id(col), val) {
            (Some(Value::Int(a)), Value::Int(b)) => a <= *b,
            (Some(Value::Float(a)), Value::Float(b)) => a <= *b,
            (Some(Value::String(a)), Value::String(b)) => a <= *b,
            _ => false,
        }
    }

    fn compare_gt(&self, row: &Row, col: &str, val: &Value) -> bool {
        match (row.get_with_id(col), val) {
            (Some(Value::Int(a)), Value::Int(b)) => a > *b,
            (Some(Value::Float(a)), Value::Float(b)) => a > *b,
            (Some(Value::String(a)), Value::String(b)) => a > *b,
            _ => false,
        }
    }

    fn compare_ge(&self, row: &Row, col: &str, val: &Value) -> bool {
        match (row.get_with_id(col), val) {
            (Some(Value::Int(a)), Value::Int(b)) => a >= *b,
            (Some(Value::Float(a)), Value::Float(b)) => a >= *b,
            (Some(Value::String(a)), Value::String(b)) => a >= *b,
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelationalError {
    TableNotFound(String),
    TableAlreadyExists(String),
    ColumnNotFound(String),
    TypeMismatch {
        column: String,
        expected: ColumnType,
    },
    NullNotAllowed(String),
    StorageError(String),
}

impl std::fmt::Display for RelationalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RelationalError::TableNotFound(t) => write!(f, "Table not found: {}", t),
            RelationalError::TableAlreadyExists(t) => write!(f, "Table already exists: {}", t),
            RelationalError::ColumnNotFound(c) => write!(f, "Column not found: {}", c),
            RelationalError::TypeMismatch { column, expected } => {
                write!(
                    f,
                    "Type mismatch for column {}: expected {:?}",
                    column, expected
                )
            },
            RelationalError::NullNotAllowed(c) => write!(f, "Null not allowed for column: {}", c),
            RelationalError::StorageError(e) => write!(f, "Storage error: {}", e),
        }
    }
}

impl std::error::Error for RelationalError {}

impl From<TensorStoreError> for RelationalError {
    fn from(e: TensorStoreError) -> Self {
        RelationalError::StorageError(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, RelationalError>;

pub struct RelationalEngine {
    store: TensorStore,
    row_counters: std::sync::RwLock<HashMap<String, AtomicU64>>,
}

impl RelationalEngine {
    pub fn new() -> Self {
        Self {
            store: TensorStore::new(),
            row_counters: std::sync::RwLock::new(HashMap::new()),
        }
    }

    pub fn with_store(store: TensorStore) -> Self {
        Self {
            store,
            row_counters: std::sync::RwLock::new(HashMap::new()),
        }
    }

    fn table_meta_key(name: &str) -> String {
        format!("_meta:table:{}", name)
    }

    fn row_key(table: &str, id: u64) -> String {
        format!("{}:{}", table, id)
    }

    fn row_prefix(table: &str) -> String {
        format!("{}:", table)
    }

    pub fn create_table(&self, name: &str, schema: Schema) -> Result<()> {
        let meta_key = Self::table_meta_key(name);

        if self.store.exists(&meta_key) {
            return Err(RelationalError::TableAlreadyExists(name.to_string()));
        }

        let mut meta = TensorData::new();
        meta.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("table".into())),
        );
        meta.set(
            "_name",
            TensorValue::Scalar(ScalarValue::String(name.into())),
        );

        let column_names: Vec<String> = schema.columns.iter().map(|c| c.name.clone()).collect();
        meta.set(
            "_columns",
            TensorValue::Scalar(ScalarValue::String(column_names.join(","))),
        );

        for col in &schema.columns {
            let type_str = match col.column_type {
                ColumnType::Int => "int",
                ColumnType::Float => "float",
                ColumnType::String => "string",
                ColumnType::Bool => "bool",
            };
            meta.set(
                format!("_col:{}", col.name),
                TensorValue::Scalar(ScalarValue::String(format!(
                    "{}:{}",
                    type_str,
                    if col.nullable { "null" } else { "notnull" }
                ))),
            );
        }

        self.store.put(meta_key, meta)?;

        let mut counters = self.row_counters.write().unwrap();
        counters.insert(name.to_string(), AtomicU64::new(0));

        Ok(())
    }

    fn get_schema(&self, table: &str) -> Result<Schema> {
        let meta_key = Self::table_meta_key(table);
        let meta = self
            .store
            .get(&meta_key)
            .map_err(|_| RelationalError::TableNotFound(table.to_string()))?;

        let columns_str = match meta.get("_columns") {
            Some(TensorValue::Scalar(ScalarValue::String(s))) => s.clone(),
            _ => return Err(RelationalError::TableNotFound(table.to_string())),
        };

        let mut columns = Vec::new();
        for col_name in columns_str.split(',') {
            if col_name.is_empty() {
                continue;
            }
            let col_key = format!("_col:{}", col_name);
            if let Some(TensorValue::Scalar(ScalarValue::String(type_str))) = meta.get(&col_key) {
                let parts: Vec<&str> = type_str.split(':').collect();
                if parts.len() == 2 {
                    let column_type = match parts[0] {
                        "int" => ColumnType::Int,
                        "float" => ColumnType::Float,
                        "string" => ColumnType::String,
                        "bool" => ColumnType::Bool,
                        _ => ColumnType::String,
                    };
                    let nullable = parts[1] == "null";
                    let mut col = Column::new(col_name, column_type);
                    if nullable {
                        col = col.nullable();
                    }
                    columns.push(col);
                }
            }
        }

        Ok(Schema::new(columns))
    }

    fn next_row_id(&self, table: &str) -> u64 {
        let counters = self.row_counters.read().unwrap();
        if let Some(counter) = counters.get(table) {
            counter.fetch_add(1, Ordering::SeqCst) + 1
        } else {
            drop(counters);
            let mut counters = self.row_counters.write().unwrap();
            let counter = counters
                .entry(table.to_string())
                .or_insert_with(|| AtomicU64::new(0));
            counter.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    pub fn insert(&self, table: &str, values: HashMap<String, Value>) -> Result<u64> {
        let schema = self.get_schema(table)?;

        for col in &schema.columns {
            let value = values.get(&col.name);
            match value {
                None | Some(Value::Null) => {
                    if !col.nullable {
                        return Err(RelationalError::NullNotAllowed(col.name.clone()));
                    }
                },
                Some(v) => {
                    if !v.matches_type(&col.column_type) {
                        return Err(RelationalError::TypeMismatch {
                            column: col.name.clone(),
                            expected: col.column_type.clone(),
                        });
                    }
                },
            }
        }

        let row_id = self.next_row_id(table);
        let key = Self::row_key(table, row_id);

        let mut tensor = TensorData::new();
        tensor.set("_id", TensorValue::Scalar(ScalarValue::Int(row_id as i64)));

        for (col_name, value) in &values {
            tensor.set(col_name, TensorValue::Scalar(value.to_scalar()));
        }

        self.store.put(key, tensor)?;
        Ok(row_id)
    }

    fn tensor_to_row(&self, tensor: &TensorData) -> Option<Row> {
        let id = match tensor.get("_id") {
            Some(TensorValue::Scalar(ScalarValue::Int(id))) => *id as u64,
            _ => return None,
        };

        let mut values = HashMap::new();
        for key in tensor.keys() {
            if key.starts_with('_') {
                continue;
            }
            if let Some(TensorValue::Scalar(scalar)) = tensor.get(key) {
                values.insert(key.clone(), Value::from_scalar(scalar));
            }
        }

        Some(Row { id, values })
    }

    pub fn select(&self, table: &str, condition: Condition) -> Result<Vec<Row>> {
        let _ = self.get_schema(table)?;

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);

        let mut rows = Vec::new();
        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(row) = self.tensor_to_row(&tensor) {
                    if condition.evaluate(&row) {
                        rows.push(row);
                    }
                }
            }
        }

        rows.sort_by_key(|r| r.id);
        Ok(rows)
    }

    pub fn update(
        &self,
        table: &str,
        condition: Condition,
        updates: HashMap<String, Value>,
    ) -> Result<usize> {
        let schema = self.get_schema(table)?;

        for (col_name, value) in &updates {
            let col = schema
                .get_column(col_name)
                .ok_or_else(|| RelationalError::ColumnNotFound(col_name.clone()))?;

            if !value.matches_type(&col.column_type) && *value != Value::Null {
                return Err(RelationalError::TypeMismatch {
                    column: col_name.clone(),
                    expected: col.column_type.clone(),
                });
            }

            if *value == Value::Null && !col.nullable {
                return Err(RelationalError::NullNotAllowed(col_name.clone()));
            }
        }

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);

        let mut count = 0;
        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(row) = self.tensor_to_row(&tensor) {
                    if condition.evaluate(&row) {
                        let mut new_tensor = tensor.clone();
                        for (col_name, value) in &updates {
                            new_tensor.set(col_name, TensorValue::Scalar(value.to_scalar()));
                        }
                        self.store.put(&key, new_tensor)?;
                        count += 1;
                    }
                }
            }
        }

        Ok(count)
    }

    pub fn delete_rows(&self, table: &str, condition: Condition) -> Result<usize> {
        let _ = self.get_schema(table)?;

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);

        let mut to_delete = Vec::new();
        for key in keys {
            if let Ok(tensor) = self.store.get(&key) {
                if let Some(row) = self.tensor_to_row(&tensor) {
                    if condition.evaluate(&row) {
                        to_delete.push(key);
                    }
                }
            }
        }

        let count = to_delete.len();
        for key in to_delete {
            self.store.delete(&key)?;
        }

        Ok(count)
    }

    pub fn join(
        &self,
        table_a: &str,
        table_b: &str,
        on_a: &str,
        on_b: &str,
    ) -> Result<Vec<(Row, Row)>> {
        let _ = self.get_schema(table_a)?;
        let _ = self.get_schema(table_b)?;

        let rows_a = self.select(table_a, Condition::True)?;
        let rows_b = self.select(table_b, Condition::True)?;

        let mut results = Vec::new();

        for row_a in &rows_a {
            let val_a = row_a.get_with_id(on_a);
            if let Some(ref va) = val_a {
                for row_b in &rows_b {
                    let val_b = row_b.get_with_id(on_b);
                    if let Some(ref vb) = val_b {
                        if va == vb {
                            results.push((row_a.clone(), row_b.clone()));
                        }
                    }
                }
            }
        }

        Ok(results)
    }

    pub fn drop_table(&self, table: &str) -> Result<()> {
        let meta_key = Self::table_meta_key(table);

        if !self.store.exists(&meta_key) {
            return Err(RelationalError::TableNotFound(table.to_string()));
        }

        let prefix = Self::row_prefix(table);
        let keys = self.store.scan(&prefix);
        for key in keys {
            self.store.delete(&key)?;
        }

        self.store.delete(&meta_key)?;

        let mut counters = self.row_counters.write().unwrap();
        counters.remove(table);

        Ok(())
    }

    pub fn table_exists(&self, table: &str) -> bool {
        let meta_key = Self::table_meta_key(table);
        self.store.exists(&meta_key)
    }

    pub fn row_count(&self, table: &str) -> Result<usize> {
        let _ = self.get_schema(table)?;
        let prefix = Self::row_prefix(table);
        Ok(self.store.scan_count(&prefix))
    }
}

impl Default for RelationalEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_users_table(engine: &RelationalEngine) {
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("age", ColumnType::Int),
            Column::new("email", ColumnType::String).nullable(),
        ]);
        engine.create_table("users", schema).unwrap();
    }

    fn create_posts_table(engine: &RelationalEngine) {
        let schema = Schema::new(vec![
            Column::new("user_id", ColumnType::Int),
            Column::new("title", ColumnType::String),
            Column::new("views", ColumnType::Int),
        ]);
        engine.create_table("posts", schema).unwrap();
    }

    #[test]
    fn create_table_and_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Alice".to_string()));
        values.insert("age".to_string(), Value::Int(30));

        let id = engine.insert("users", values).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn insert_1000_rows_select_with_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..1000 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + (i % 50)));
            engine.insert("users", values).unwrap();
        }

        assert_eq!(engine.row_count("users").unwrap(), 1000);

        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();

        assert_eq!(rows.len(), 20);

        for row in &rows {
            assert_eq!(row.get("age"), Some(&Value::Int(25)));
        }
    }

    #[test]
    fn select_with_range_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Ge("age".to_string(), Value::Int(90)))
            .unwrap();

        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn select_with_compound_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..100 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let condition = Condition::Ge("age".to_string(), Value::Int(40))
            .and(Condition::Lt("age".to_string(), Value::Int(50)));

        let rows = engine.select("users", condition).unwrap();
        assert_eq!(rows.len(), 10);
    }

    #[test]
    fn join_two_tables() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_posts_table(&engine);

        for i in 1..=5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let post_data = vec![
            (1, "Post A", 100),
            (1, "Post B", 200),
            (2, "Post C", 150),
            (3, "Post D", 50),
            (3, "Post E", 75),
            (3, "Post F", 25),
        ];

        for (user_id, title, views) in post_data {
            let mut values = HashMap::new();
            values.insert("user_id".to_string(), Value::Int(user_id));
            values.insert("title".to_string(), Value::String(title.to_string()));
            values.insert("views".to_string(), Value::Int(views));
            engine.insert("posts", values).unwrap();
        }

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();

        assert_eq!(joined.len(), 6);

        let user1_posts: Vec<_> = joined.iter().filter(|(u, _)| u.id == 1).collect();
        assert_eq!(user1_posts.len(), 2);

        let user3_posts: Vec<_> = joined.iter().filter(|(u, _)| u.id == 3).collect();
        assert_eq!(user3_posts.len(), 3);
    }

    #[test]
    fn update_modifies_correct_rows() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(25));
            engine.insert("users", values).unwrap();
        }

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::Int(30));

        let count = engine
            .update(
                "users",
                Condition::Lt("_id".to_string(), Value::Int(6)),
                updates,
            )
            .unwrap();

        assert_eq!(count, 5);

        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(30)))
            .unwrap();
        assert_eq!(rows.len(), 5);

        let rows = engine
            .select("users", Condition::Eq("age".to_string(), Value::Int(25)))
            .unwrap();
        assert_eq!(rows.len(), 5);
    }

    #[test]
    fn delete_removes_correct_rows() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..20 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        assert_eq!(engine.row_count("users").unwrap(), 20);

        let count = engine
            .delete_rows("users", Condition::Lt("age".to_string(), Value::Int(10)))
            .unwrap();

        assert_eq!(count, 10);
        assert_eq!(engine.row_count("users").unwrap(), 10);

        let remaining = engine.select("users", Condition::True).unwrap();
        for row in remaining {
            if let Some(Value::Int(age)) = row.get("age") {
                assert!(*age >= 10);
            }
        }
    }

    #[test]
    fn delete_data_is_gone() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("ToDelete".to_string()));
        values.insert("age".to_string(), Value::Int(99));
        let id = engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(id as i64)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);

        engine
            .delete_rows(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(id as i64)),
            )
            .unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Eq("_id".to_string(), Value::Int(id as i64)),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn table_not_found_error() {
        let engine = RelationalEngine::new();

        let result = engine.select("nonexistent", Condition::True);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn duplicate_table_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let schema = Schema::new(vec![Column::new("x", ColumnType::Int)]);
        let result = engine.create_table("users", schema);
        assert!(matches!(
            result,
            Err(RelationalError::TableAlreadyExists(_))
        ));
    }

    #[test]
    fn type_mismatch_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::Int(123));
        values.insert("age".to_string(), Value::Int(30));

        let result = engine.insert("users", values);
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
    }

    #[test]
    fn null_not_allowed_error() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::Null);
        values.insert("age".to_string(), Value::Int(30));

        let result = engine.insert("users", values);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
    }

    #[test]
    fn nullable_column_accepts_null() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        values.insert("email".to_string(), Value::Null);

        let id = engine.insert("users", values).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn drop_table() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".to_string()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        assert!(engine.table_exists("users"));

        engine.drop_table("users").unwrap();

        assert!(!engine.table_exists("users"));

        let result = engine.select("users", Condition::True);
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn or_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let condition = Condition::Eq("age".to_string(), Value::Int(0))
            .or(Condition::Eq("age".to_string(), Value::Int(9)));

        let rows = engine.select("users", condition).unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn row_id_in_condition() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(20 + i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Eq("_id".to_string(), Value::Int(3)))
            .unwrap();

        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].id, 3);
    }

    // Additional tests for 100% coverage

    #[test]
    fn error_display_all_variants() {
        let e1 = RelationalError::TableNotFound("test".into());
        assert!(format!("{}", e1).contains("test"));

        let e2 = RelationalError::TableAlreadyExists("test".into());
        assert!(format!("{}", e2).contains("test"));

        let e3 = RelationalError::ColumnNotFound("col".into());
        assert!(format!("{}", e3).contains("col"));

        let e4 = RelationalError::TypeMismatch {
            column: "age".into(),
            expected: ColumnType::Int,
        };
        assert!(format!("{}", e4).contains("age"));

        let e5 = RelationalError::NullNotAllowed("name".into());
        assert!(format!("{}", e5).contains("name"));

        let e6 = RelationalError::StorageError("disk full".into());
        assert!(format!("{}", e6).contains("disk full"));
    }

    #[test]
    fn error_is_error_trait() {
        let err: &dyn std::error::Error = &RelationalError::TableNotFound("x".into());
        assert!(err.to_string().contains("x"));
    }

    #[test]
    fn engine_default_trait() {
        let engine = RelationalEngine::default();
        assert!(!engine.table_exists("any"));
    }

    #[test]
    fn engine_with_store() {
        let store = TensorStore::new();
        let engine = RelationalEngine::with_store(store);
        assert!(!engine.table_exists("any"));
    }

    #[test]
    fn row_get_returns_none_for_id() {
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("test".into()));
        let row = Row { id: 1, values };
        assert!(row.get("_id").is_none());
        assert_eq!(row.get_with_id("_id"), Some(Value::Int(1)));
    }

    #[test]
    fn condition_ne() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..5 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Ne("age".to_string(), Value::Int(2)))
            .unwrap();
        assert_eq!(rows.len(), 4);
    }

    #[test]
    fn condition_le() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Le("age".to_string(), Value::Int(5)))
            .unwrap();
        assert_eq!(rows.len(), 6);
    }

    #[test]
    fn condition_gt() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("age".to_string(), Value::Int(i));
            engine.insert("users", values).unwrap();
        }

        let rows = engine
            .select("users", Condition::Gt("age".to_string(), Value::Int(7)))
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[test]
    fn float_comparisons() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("score", ColumnType::Float),
        ]);
        engine.create_table("scores", schema).unwrap();

        for i in 0..10 {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(format!("User{}", i)));
            values.insert("score".to_string(), Value::Float(i as f64 * 0.5));
            engine.insert("scores", values).unwrap();
        }

        let lt = engine
            .select(
                "scores",
                Condition::Lt("score".to_string(), Value::Float(2.0)),
            )
            .unwrap();
        assert_eq!(lt.len(), 4);

        let le = engine
            .select(
                "scores",
                Condition::Le("score".to_string(), Value::Float(2.0)),
            )
            .unwrap();
        assert_eq!(le.len(), 5);

        let gt = engine
            .select(
                "scores",
                Condition::Gt("score".to_string(), Value::Float(3.5)),
            )
            .unwrap();
        assert_eq!(gt.len(), 2);

        let ge = engine
            .select(
                "scores",
                Condition::Ge("score".to_string(), Value::Float(3.5)),
            )
            .unwrap();
        assert_eq!(ge.len(), 3);
    }

    #[test]
    fn string_comparisons() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let names = vec!["Alice", "Bob", "Charlie", "David", "Eve"];
        for name in &names {
            let mut values = HashMap::new();
            values.insert("name".to_string(), Value::String(name.to_string()));
            values.insert("age".to_string(), Value::Int(30));
            engine.insert("users", values).unwrap();
        }

        let lt = engine
            .select(
                "users",
                Condition::Lt("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(lt.len(), 2);

        let le = engine
            .select(
                "users",
                Condition::Le("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(le.len(), 3);

        let gt = engine
            .select(
                "users",
                Condition::Gt("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(gt.len(), 2);

        let ge = engine
            .select(
                "users",
                Condition::Ge("name".to_string(), Value::String("Charlie".into())),
            )
            .unwrap();
        assert_eq!(ge.len(), 3);
    }

    #[test]
    fn update_column_not_found() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let mut updates = HashMap::new();
        updates.insert("nonexistent".to_string(), Value::Int(1));

        let result = engine.update("users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::ColumnNotFound(_))));
    }

    #[test]
    fn update_type_mismatch() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let mut updates = HashMap::new();
        updates.insert("age".to_string(), Value::String("wrong type".into()));

        let result = engine.update("users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::TypeMismatch { .. })));
    }

    #[test]
    fn update_null_not_allowed() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let mut updates = HashMap::new();
        updates.insert("name".to_string(), Value::Null);

        let result = engine.update("users", Condition::True, updates);
        assert!(matches!(result, Err(RelationalError::NullNotAllowed(_))));
    }

    #[test]
    fn drop_nonexistent_table() {
        let engine = RelationalEngine::new();
        let result = engine.drop_table("nonexistent");
        assert!(matches!(result, Err(RelationalError::TableNotFound(_))));
    }

    #[test]
    fn join_no_matches() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);
        create_posts_table(&engine);

        let mut user_values = HashMap::new();
        user_values.insert("name".to_string(), Value::String("Alice".into()));
        user_values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", user_values).unwrap();

        let mut post_values = HashMap::new();
        post_values.insert("user_id".to_string(), Value::Int(999));
        post_values.insert("title".to_string(), Value::String("Orphan".into()));
        post_values.insert("views".to_string(), Value::Int(0));
        engine.insert("posts", post_values).unwrap();

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();
        assert_eq!(joined.len(), 0);
    }

    #[test]
    fn empty_table_select() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let rows = engine.select("users", Condition::True).unwrap();
        assert!(rows.is_empty());
    }

    #[test]
    fn value_clone_and_eq() {
        let v1 = Value::Null;
        let v2 = Value::Int(42);
        let v3 = Value::Float(3.14);
        let v4 = Value::String("test".into());
        let v5 = Value::Bool(true);

        assert_eq!(v1.clone(), v1);
        assert_eq!(v2.clone(), v2);
        assert_eq!(v3.clone(), v3);
        assert_eq!(v4.clone(), v4);
        assert_eq!(v5.clone(), v5);
    }

    #[test]
    fn column_type_clone_and_eq() {
        assert_eq!(ColumnType::Int.clone(), ColumnType::Int);
        assert_eq!(ColumnType::Float.clone(), ColumnType::Float);
        assert_eq!(ColumnType::String.clone(), ColumnType::String);
        assert_eq!(ColumnType::Bool.clone(), ColumnType::Bool);
    }

    #[test]
    fn schema_get_column() {
        let schema = Schema::new(vec![
            Column::new("id", ColumnType::Int),
            Column::new("name", ColumnType::String),
        ]);

        assert!(schema.get_column("id").is_some());
        assert!(schema.get_column("name").is_some());
        assert!(schema.get_column("nonexistent").is_none());
    }

    #[test]
    fn value_from_bytes_scalar() {
        let bytes_scalar = ScalarValue::Bytes(vec![1, 2, 3]);
        let value = Value::from_scalar(&bytes_scalar);
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn condition_debug() {
        let c = Condition::True;
        let debug_str = format!("{:?}", c);
        assert!(debug_str.contains("True"));
    }

    #[test]
    fn row_debug_and_clone() {
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("test".into()));
        let row = Row { id: 1, values };
        let cloned = row.clone();
        assert_eq!(cloned.id, 1);
        let debug_str = format!("{:?}", row);
        assert!(debug_str.contains("Row"));
    }

    #[test]
    fn column_debug_and_clone() {
        let col = Column::new("test", ColumnType::Int);
        let cloned = col.clone();
        assert_eq!(cloned.name, "test");
        let debug_str = format!("{:?}", col);
        assert!(debug_str.contains("Column"));
    }

    #[test]
    fn schema_debug_and_clone() {
        let schema = Schema::new(vec![Column::new("id", ColumnType::Int)]);
        let cloned = schema.clone();
        assert_eq!(cloned.columns.len(), 1);
        let debug_str = format!("{:?}", schema);
        assert!(debug_str.contains("Schema"));
    }

    #[test]
    fn error_clone_and_eq() {
        let e1 = RelationalError::TableNotFound("test".into());
        let e2 = RelationalError::TableAlreadyExists("test".into());
        let e3 = RelationalError::ColumnNotFound("col".into());
        let e4 = RelationalError::TypeMismatch {
            column: "age".into(),
            expected: ColumnType::Int,
        };
        let e5 = RelationalError::NullNotAllowed("name".into());
        let e6 = RelationalError::StorageError("err".into());

        assert_eq!(e1.clone(), e1);
        assert_eq!(e2.clone(), e2);
        assert_eq!(e3.clone(), e3);
        assert_eq!(e4.clone(), e4);
        assert_eq!(e5.clone(), e5);
        assert_eq!(e6.clone(), e6);
    }

    #[test]
    fn storage_error_from_tensor_store() {
        use tensor_store::TensorStoreError;
        let tensor_err = TensorStoreError::NotFound("key".into());
        let rel_err: RelationalError = tensor_err.into();
        assert!(matches!(rel_err, RelationalError::StorageError(_)));
    }

    #[test]
    fn insert_missing_nullable_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        // email is nullable and not provided

        let id = engine.insert("users", values).unwrap();
        assert!(id > 0);
    }

    #[test]
    fn comparison_with_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        // Compare int column with string value - should match nothing
        let rows = engine
            .select(
                "users",
                Condition::Lt("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn comparison_with_null_column_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        values.insert("email".to_string(), Value::Null);
        engine.insert("users", values).unwrap();

        // Comparing null email with string - should return false
        let rows = engine
            .select(
                "users",
                Condition::Lt("email".to_string(), Value::String("z".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn bool_column_type() {
        let engine = RelationalEngine::new();
        let schema = Schema::new(vec![
            Column::new("name", ColumnType::String),
            Column::new("active", ColumnType::Bool),
        ]);
        engine.create_table("flags", schema).unwrap();

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("active".to_string(), Value::Bool(true));
        let id = engine.insert("flags", values).unwrap();
        assert!(id > 0);

        let rows = engine
            .select(
                "flags",
                Condition::Eq("active".to_string(), Value::Bool(true)),
            )
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[test]
    fn row_counter_initialization_on_insert() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        // Drop and recreate to test counter reinitialization path
        engine.drop_table("users").unwrap();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        let id = engine.insert("users", values).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn value_debug() {
        let v = Value::Int(42);
        let debug_str = format!("{:?}", v);
        assert!(debug_str.contains("Int"));
    }

    #[test]
    fn column_type_debug() {
        let ct = ColumnType::Float;
        let debug_str = format!("{:?}", ct);
        assert!(debug_str.contains("Float"));
    }

    #[test]
    fn condition_clone() {
        let c1 = Condition::Eq("col".into(), Value::Int(1));
        let c2 = Condition::Ne("col".into(), Value::Int(2));
        let c3 = Condition::Lt("col".into(), Value::Int(3));
        let c4 = Condition::Le("col".into(), Value::Int(4));
        let c5 = Condition::Gt("col".into(), Value::Int(5));
        let c6 = Condition::Ge("col".into(), Value::Int(6));
        let c7 = Condition::True;

        let _ = c1.clone();
        let _ = c2.clone();
        let _ = c3.clone();
        let _ = c4.clone();
        let _ = c5.clone();
        let _ = c6.clone();
        let _ = c7.clone();

        let c8 = Condition::And(Box::new(Condition::True), Box::new(Condition::True));
        let c9 = Condition::Or(Box::new(Condition::True), Box::new(Condition::True));
        let _ = c8.clone();
        let _ = c9.clone();
    }

    #[test]
    fn row_get_nonexistent_column() {
        let row = Row {
            id: 1,
            values: HashMap::new(),
        };
        assert!(row.get("nonexistent").is_none());
        assert!(row.get_with_id("nonexistent").is_none());
    }

    #[test]
    fn compare_le_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Le("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn compare_gt_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Gt("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn compare_ge_mismatched_types_returns_false() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", values).unwrap();

        let rows = engine
            .select(
                "users",
                Condition::Ge("age".to_string(), Value::String("30".into())),
            )
            .unwrap();
        assert_eq!(rows.len(), 0);
    }

    #[test]
    fn next_row_id_without_counter_initialized() {
        // Use with_store to create engine without going through create_table
        let store = TensorStore::new();

        // Manually insert table metadata without initializing the counter
        let mut meta = TensorData::new();
        meta.set(
            "_type",
            TensorValue::Scalar(ScalarValue::String("table".into())),
        );
        meta.set(
            "_name",
            TensorValue::Scalar(ScalarValue::String("manual_table".into())),
        );
        meta.set(
            "_columns",
            TensorValue::Scalar(ScalarValue::String("name,age".into())),
        );
        meta.set(
            "_col:name",
            TensorValue::Scalar(ScalarValue::String("string:notnull".into())),
        );
        meta.set(
            "_col:age",
            TensorValue::Scalar(ScalarValue::String("int:notnull".into())),
        );
        store.put("_meta:table:manual_table", meta).unwrap();

        let engine = RelationalEngine::with_store(store);

        // Now insert - this should trigger the else branch in next_row_id
        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        let id = engine.insert("manual_table", values).unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn update_with_nullable_null_value() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let mut values = HashMap::new();
        values.insert("name".to_string(), Value::String("Test".into()));
        values.insert("age".to_string(), Value::Int(30));
        values.insert("email".to_string(), Value::String("test@test.com".into()));
        engine.insert("users", values).unwrap();

        // Update email (nullable) to Null - should succeed
        let mut updates = HashMap::new();
        updates.insert("email".to_string(), Value::Null);
        let count = engine.update("users", Condition::True, updates).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn join_with_null_join_column() {
        let engine = RelationalEngine::new();
        create_users_table(&engine);

        let schema = Schema::new(vec![
            Column::new("user_id", ColumnType::Int).nullable(),
            Column::new("title", ColumnType::String),
        ]);
        engine.create_table("posts", schema).unwrap();

        // Insert user
        let mut user_values = HashMap::new();
        user_values.insert("name".to_string(), Value::String("Alice".into()));
        user_values.insert("age".to_string(), Value::Int(30));
        engine.insert("users", user_values).unwrap();

        // Insert post with null user_id
        let mut post_values = HashMap::new();
        post_values.insert("user_id".to_string(), Value::Null);
        post_values.insert("title".to_string(), Value::String("Orphan".into()));
        engine.insert("posts", post_values).unwrap();

        let joined = engine.join("users", "posts", "_id", "user_id").unwrap();
        // Should not match because null != 1
        assert_eq!(joined.len(), 0);
    }
}

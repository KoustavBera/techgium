import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/report_metadata.dart';

class DatabaseService {
  static final DatabaseService _instance = DatabaseService._internal();
  factory DatabaseService() => _instance;
  DatabaseService._internal();

  static Database? _database;

  Future<Database> get database async {
    _database ??= await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'chiranjeevi.db');
    return openDatabase(
      path,
      version: 1,
      onCreate: (db, version) async {
        await db.execute('''
          CREATE TABLE reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_id TEXT UNIQUE NOT NULL,
            patient_name TEXT,
            downloaded_at TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_size_bytes INTEGER,
            source_url TEXT
          )
        ''');
      },
    );
  }

  Future<int> insertReport(ReportMetadata report) async {
    final db = await database;
    return await db.insert(
      'reports',
      report.toMap(),
      conflictAlgorithm: ConflictAlgorithm.ignore,
    );
  }

  Future<List<ReportMetadata>> getAllReports() async {
    final db = await database;
    final List<Map<String, dynamic>> maps = await db.query(
      'reports',
      orderBy: 'downloaded_at DESC',
    );
    return List.generate(maps.length, (i) => ReportMetadata.fromMap(maps[i]));
  }

  Future<ReportMetadata?> getReportById(String reportId) async {
    final db = await database;
    final maps = await db.query(
      'reports',
      where: 'report_id = ?',
      whereArgs: [reportId],
    );
    if (maps.isNotEmpty) {
      return ReportMetadata.fromMap(maps.first);
    }
    return null;
  }

  Future<void> deleteReport(String reportId) async {
    final db = await database;
    await db.delete(
      'reports',
      where: 'report_id = ?',
      whereArgs: [reportId],
    );
  }

  Future<bool> reportExists(String reportId) async {
    final db = await database;
    final maps = await db.query(
      'reports',
      columns: ['id'],
      where: 'report_id = ?',
      whereArgs: [reportId],
      limit: 1,
    );
    return maps.isNotEmpty;
  }
}

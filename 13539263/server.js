const express = require('express');
const multer = require('multer');
const path = require('path');
const jwt = require('jsonwebtoken');
const fs = require('fs').promises;
const app = express();
const pdfParse = require('pdf-parse');
const mammoth = require('mammoth');

// JWT 密钥
const JWT_SECRET = 'your-secret-key';

// 用户数据文件路径
const USER_DATA_FILE = path.join(__dirname, 'data', 'users.json');

// 作业数据文件路径
const SUBMISSIONS_FILE = path.join(__dirname, 'data', 'submissions.json');
const GRADED_SUBMISSIONS_FILE = path.join(__dirname, 'data', 'graded_submissions.json');

// 读取用户数据
async function loadUsers() {
    try {
        const data = await fs.readFile(USER_DATA_FILE, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        // 如果文件不存在，返回默认用户数据
        return {
            teachers: [
                { username: 'teacher', password: '123456', name: '教师1' }
            ],
            students: [
                { username: 'student1', password: '123456', name: '张三' }
            ]
        };
    }
}

// 保存用户数据
async function saveUsers(users) {
    try {
        // 确保目录存在
        await fs.mkdir(path.dirname(USER_DATA_FILE), { recursive: true });
        // 保存数据
        await fs.writeFile(USER_DATA_FILE, JSON.stringify(users, null, 4));
    } catch (error) {
        console.error('保存用户数据失败:', error);
        throw error;
    }
}

// 读取作业数据
async function loadSubmissions() {
    try {
        const submissionsData = await fs.readFile(SUBMISSIONS_FILE, 'utf8');
        const gradedData = await fs.readFile(GRADED_SUBMISSIONS_FILE, 'utf8');
        return {
            submissions: JSON.parse(submissionsData),
            gradedSubmissions: JSON.parse(gradedData)
        };
    } catch (error) {
        // 如果文件不存在，返回空数组
        return {
            submissions: [],
            gradedSubmissions: []
        };
    }
}

// 保存作业数据
async function saveSubmissions() {
    try {
        // 确保目录存在
        await fs.mkdir(path.dirname(SUBMISSIONS_FILE), { recursive: true });
        // 保存数据
        await fs.writeFile(SUBMISSIONS_FILE, JSON.stringify(submissions, null, 4));
        await fs.writeFile(GRADED_SUBMISSIONS_FILE, JSON.stringify(gradedSubmissions, null, 4));
    } catch (error) {
        console.error('保存作业数据失败:', error);
        throw error;
    }
}

// 初始化用户数据
let users = {
    teachers: [],
    students: []
};

// 初始化作业数据
let submissions = [];
let gradedSubmissions = [];

// 服务器启动时加载用户数据
(async () => {
    try {
        users = await loadUsers();
        console.log('用户数据加载成功');
    } catch (error) {
        console.error('加载用户数据失败:', error);
    }
})();

// 服务器启动时加载作业数据
(async () => {
    try {
        const data = await loadSubmissions();
        submissions = data.submissions;
        gradedSubmissions = data.gradedSubmissions;
        console.log('作业数据加载成功');
    } catch (error) {
        console.error('加载作业数据失败:', error);
    }
})();

// 配置跨域
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
    if (req.method === 'OPTIONS') {
        return res.sendStatus(200);
    }
    next();
});

// 基础中间件
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));
app.use('/uploads', express.static(path.join(__dirname, 'uploads')));

// 验证 token 的中间件
const authenticateToken = (req, res, next) => {
    // 不需要验证的路由
    const publicPaths = ['/api/login', '/', '/login.html', '/css/style.css', '/js/login.js'];
    if (publicPaths.includes(req.path) || req.path.startsWith('/uploads/')) {
        return next();
    }

    const authHeader = req.headers['authorization'];
    const token = authHeader && authHeader.split(' ')[1];

    if (!token) {
        return res.status(401).json({ success: false, message: '未登录' });
    }

    jwt.verify(token, JWT_SECRET, (err, user) => {
        if (err) {
            return res.status(403).json({ success: false, message: '登录已过期' });
        }
        req.user = user;
        next();
    });
};

// 应用认证中间件
app.use(authenticateToken);

// 登录路由
app.post('/api/login', (req, res) => {
    const { username, password, type } = req.body;
    const userList = type === 'teacher' ? users.teachers : users.students;
    
    const user = userList.find(u => u.username === username && u.password === password);
    
    if (user) {
        const token = jwt.sign({ 
            username: user.username,
            type: type,
            name: user.name
        }, JWT_SECRET, { expiresIn: '24h' });
        
        res.json({ 
            success: true, 
            token,
            user: {
                username: user.username,
                name: user.name,
                type: type
            }
        });
    } else {
        res.status(401).json({ success: false, message: '用户名或密码错误' });
    }
});

// 添加学生账号
app.post('/api/students', authenticateToken, async (req, res) => {
    if (req.user.type !== 'teacher') {
        return res.status(403).json({ success: false, message: '无权限' });
    }

    const { username, password, name } = req.body;
    
    if (users.students.some(s => s.username === username)) {
        return res.status(400).json({ success: false, message: '用户名已存在' });
    }

    users.students.push({ username, password, name });
    
    try {
        await saveUsers(users);
        res.json({ success: true });
    } catch (error) {
        console.error('保存用户数据失败:', error);
        res.status(500).json({ success: false, message: '服务器错误' });
    }
});

// 获取学生列表
app.get('/api/students', authenticateToken, (req, res) => {
    if (req.user.type !== 'teacher') {
        return res.status(403).json({ success: false, message: '无权限' });
    }

    const studentList = users.students.map(s => ({
        username: s.username,
        name: s.name
    }));
    
    res.json(studentList);
});

// 删除学生账号
app.delete('/api/students/:username', authenticateToken, async (req, res) => {
    if (req.user.type !== 'teacher') {
        return res.status(403).json({ success: false, message: '无权限' });
    }

    const username = req.params.username;
    users.students = users.students.filter(s => s.username !== username);
    
    try {
        await saveUsers(users);
        res.json({ success: true });
    } catch (error) {
        console.error('保存用户数据失败:', error);
        res.status(500).json({ success: false, message: '服务器错误' });
    }
});

// 配置文件上传
const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'uploads/')
    },
    filename: function (req, file, cb) {
        cb(null, Date.now() + '-' + file.originalname)
    }
});
const upload = multer({ storage: storage });

// 学生提交作业
app.post('/api/submit', authenticateToken, upload.single('file'), async (req, res) => {
    if (req.user.type !== 'student') {
        return res.status(403).json({ 
            success: false, 
            message: '只有学生可以提交作业' 
        });
    }

    const { studentName, studentId } = req.body;
    const file = req.file;
    
    // 提取文本内容
    let text = '';
    try {
        if (file.mimetype === 'application/pdf') {
            text = await extractTextFromPDF(file.path);
        } else if (file.mimetype.includes('word')) {
            text = await extractTextFromWord(file.path);
        }
    } catch (error) {
        console.error('文本提取失败:', error);
    }
    
    const submission = {
        id: submissions.length + gradedSubmissions.length + 1,
        studentName,
        studentId,
        fileName: file.originalname,
        fileUrl: `/uploads/${file.filename}`,
        submitTime: new Date().toLocaleString(),
        status: '待批改',
        text: text
    };
    
    submissions.push(submission);
    
    // 保存到文件
    try {
        await saveSubmissions();
        res.json({ success: true, submission });
    } catch (error) {
        console.error('保存作业数据失败:', error);
        res.status(500).json({ success: false, message: '服务器错误' });
    }
});

// 获取所有待批改作业
app.get('/api/submissions', authenticateToken, (req, res) => {
    if (req.user.type !== 'teacher') {
        return res.status(403).json({ 
            success: false, 
            message: '只有教师可以查看待批改作业' 
        });
    }
    const pendingSubmissions = submissions.filter(s => s.status === '待批改');
    res.json(pendingSubmissions);
});

// 获取所有已批改作业
app.get('/api/graded-submissions', authenticateToken, (req, res) => {
    if (req.user.type !== 'teacher') {
        return res.status(403).json({ 
            success: false, 
            message: '只有教师可以查看已批改作业' 
        });
    }
    res.json(gradedSubmissions);
});

// 提交评分
app.post('/api/grade', authenticateToken, async (req, res) => {
    if (req.user.type !== 'teacher') {
        return res.status(403).json({ 
            success: false, 
            message: '只有教师可以评分' 
        });
    }

    const { submissionId, score, feedback } = req.body;
    const submission = submissions.find(s => s.id === parseInt(submissionId));
    
    if (submission) {
        submission.score = score;
        submission.feedback = feedback;
        submission.status = '已批改';
        submission.gradedTime = new Date().toLocaleString();

        gradedSubmissions.push(submission);
        submissions = submissions.filter(s => s.id !== parseInt(submissionId));

        try {
            await saveSubmissions();
            res.json({ success: true });
        } catch (error) {
            console.error('保存评分数据失败:', error);
            res.status(500).json({ success: false, message: '服务器错误' });
        }
    } else {
        res.status(404).json({ success: false, message: '未找到该作业' });
    }
});

// 获取学生提交历史
app.get('/api/history/:studentId', (req, res) => {
    const studentSubmissions = [
        ...submissions.filter(s => s.studentId === req.params.studentId),
        ...gradedSubmissions.filter(s => s.studentId === req.params.studentId)
    ];
    res.json(studentSubmissions);
});

// 错误处理中间件
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('服务器错误！');
});

// 处理 404
app.use((req, res) => {
    res.status(404).send('未找到请求的资源');
});

async function extractTextFromPDF(filePath) {
    const dataBuffer = await fs.readFile(filePath);
    const data = await pdfParse(dataBuffer);
    return data.text;
}

async function extractTextFromWord(filePath) {
    const result = await mammoth.extractRawText({path: filePath});
    return result.value;
}

const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
}); 
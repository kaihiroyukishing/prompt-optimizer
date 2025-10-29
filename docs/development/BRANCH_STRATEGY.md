# Branch Strategy and Protection

## Branch Types

### `main`
- **Purpose**: Production-ready code
- **Protection**: Full protection enabled
- **Requirements**:
  - All CI checks must pass
  - At least 1 approving review
  - No force pushes
  - No direct commits

### `develop`
- **Purpose**: Integration branch for development
- **Protection**: Moderate protection
- **Requirements**:
  - All CI checks must pass
  - No force pushes

### `feature/*`
- **Purpose**: New features and enhancements
- **Branch from**: `develop`
- **Merge to**: `develop`
- **Examples**: `feature/user-authentication`, `feature/cache-implementation`

### `bugfix/*`
- **Purpose**: Bug fixes
- **Branch from**: `develop` (or `main` for hotfixes)
- **Merge to**: `develop` (or `main` for hotfixes)
- **Examples**: `bugfix/rate-limit-issue`, `bugfix/database-connection`

### `hotfix/*`
- **Purpose**: Critical production fixes
- **Branch from**: `main`
- **Merge to**: `main` and `develop`
- **Examples**: `hotfix/security-patch`, `hotfix/critical-bug`

### `refactor/*`
- **Purpose**: Code refactoring
- **Branch from**: `develop`
- **Merge to**: `develop`
- **Examples**: `refactor/backend-structure`, `refactor/api-routes`

### `docs/*`
- **Purpose**: Documentation updates
- **Branch from**: `develop`
- **Merge to**: `develop`
- **Examples**: `docs/api-documentation`, `docs/setup-guide`

## Branch Protection Rules

### For `main` Branch
1. **Require pull request reviews**
   - Minimum: 1 approving review
   - Dismiss stale reviews on new commits: Yes

2. **Require status checks**
   - All CI checks must pass
   - Require branches to be up to date before merging

3. **Require conversation resolution**
   - All conversations must be resolved before merging

4. **Restrictions**
   - No force pushes
   - No branch deletion
   - Limit to collaborators only

### For `develop` Branch
1. **Require pull request reviews**
   - Minimum: 1 approving review

2. **Require status checks**
   - All CI checks must pass

3. **Restrictions**
   - No force pushes

## Pull Request Workflow

1. **Create Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write your code
   - Add tests
   - Update documentation

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commit format:
   - `feat:` - New feature
   - `fix:` - Bug fix
   - `docs:` - Documentation
   - `refactor:` - Code refactoring
   - `test:` - Tests
   - `chore:` - Maintenance

4. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a PR on GitHub from your branch to `develop`

5. **Code Review**
   - Address review comments
   - Push additional commits if needed
   - Ensure all CI checks pass

6. **Merge**
   - Squash and merge (preferred for feature branches)
   - Delete branch after merging

## Merging Strategies

### Feature/Bugfix/Refactor Branches
- **Strategy**: Squash and merge
- **Reason**: Keeps history clean with single descriptive commit

### Hotfix Branches
- **Strategy**: Merge commit
- **Reason**: Preserves hotfix history

### Documentation Branches
- **Strategy**: Merge commit or squash
- **Reason**: Flexibility based on PR size

## Code Review Guidelines

### For Authors
- Keep PRs small and focused
- Write clear PR descriptions
- Make sure all tests pass
- Update documentation as needed
- Be responsive to feedback

### For Reviewers
- Be constructive and respectful
- Focus on code quality and correctness
- Check for security issues
- Ensure tests are adequate
- Verify documentation is updated

## Release Process

1. **Merge to `main`**: PR from `develop` to `main`
2. **Tag Release**: Create tag with version number
3. **Deploy**: Deploy to production
4. **Hotfix**: Create hotfix branch from `main` if needed

## Emergency Hotfix Process

1. Create hotfix branch from `main`
2. Implement fix
3. Create PR to `main` (urgent review)
4. Merge to `main` and deploy
5. Merge back to `develop`
6. Delete hotfix branch

---

**Note**: These rules should be configured in the GitHub repository settings under "Branches".
